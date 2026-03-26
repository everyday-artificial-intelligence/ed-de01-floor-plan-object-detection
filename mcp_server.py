import io
import json
import logging
import os
import tempfile
import time

# MCP stdio servers must keep stdout clean; configure writable cache dirs up front
# so library startup does not emit warnings about non-writable defaults.
RUNTIME_DIR = os.path.join(tempfile.gettempdir(), "floor-plan-door-detection")
MPLCONFIG_DIR = os.path.join(RUNTIME_DIR, "matplotlib")
YOLO_CONFIG_DIR = os.path.join(RUNTIME_DIR, "ultralytics")
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)

import httpx
import PIL.Image
import PIL.ImageFile

# Floor plans can be very large; raise PIL's decompression bomb limit so they
# load without warnings or errors. The images come from our own pipeline so
# the DOS-guard is not needed here.
PIL.Image.MAX_IMAGE_PIXELS = None
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
import torch
from ultralytics import YOLO

# PyTorch 2.6+ changed torch.load to default weights_only=True, which blocks
# loading YOLO checkpoints (they contain custom classes not on the safe list).
# Patch torch.load so callers that don't specify weights_only (i.e. ultralytics)
# fall back to False — safe here because best.pt is our own trusted model file.
_orig_torch_load = torch.load
def _torch_load_compat(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_compat

load_dotenv()

import google.cloud.logging as gcp_logging
gcp_logging.Client().setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Constants (mirrored from app.py) ---
DOOR_DERP_PROMPT = (
    "You will receive multiple door crop images.\n"
    "Each image is preceded by a text part formatted exactly as 'door_id: <value>'.\n"
    "Return JSON only and follow the provided schema exactly.\n"
    "In some cases the crops may not contain a door if that is the case return an object with unknown values for both fields immediately.\n"
    "If you are even the slightest bit unsure about any fields mark them as needing review.\n"
    "Return one object for every supplied door crop, in the same order, reusing the exact door_id value.\n\n"

    "For each door determine:\n"
    "1. opening_side: which side of the crop contains the door opening relative to the image orientation. "
    "Use one of top, left, right, bottom, or unknown. "
    "The opening will be an open gap between lines or maybe a thin line. "
    "Two lines or a thick line signifies a wall, not the opening."
    "If it is not clear which side the opening is on as multiple sides seem possible, mark opening_side as unknown.\n"
    "2. arc_type: which clock-face quarter the arc sweeps through. "
    "Use one of 12_to_3, 3_to_6, 6_to_9, 9_to_12, or unknown.\n\n"
    "If a door is ambiguous, include it and use unknown for the uncertain field.\n"
    "If two doors are present in a single door crop, mark both fields as unknown. Apply this if even a tiny bit of another door is visible."
)
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_CONNECT_TIMEOUT_S = 10.0
GEMINI_READ_TIMEOUT_S = 90.0
GEMINI_WRITE_TIMEOUT_S = 30.0
GEMINI_POOL_TIMEOUT_S = 30.0
GEMINI_CONNECT_RETRIES = 0
GEMINI_DOOR_BATCH_SIZE = 5
OPENING_SIDE_VALUES = ("top", "left", "right", "bottom", "unknown")
ARC_TYPE_VALUES = ("12_to_3", "3_to_6", "6_to_9", "9_to_12", "unknown")
DOOR_ANALYSIS_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["doors"],
    properties={
        "doors": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["door_id", "opening_side", "arc_type"],
                properties={
                    "door_id": types.Schema(type=types.Type.STRING),
                    "opening_side": types.Schema(
                        type=types.Type.STRING,
                        enum=list(OPENING_SIDE_VALUES),
                    ),
                    "arc_type": types.Schema(
                        type=types.Type.STRING,
                        enum=list(ARC_TYPE_VALUES),
                    ),
                },
            ),
        ),
    },
)

DERP_MAP = {
    ("12_to_3", "left"): "LH",
    ("12_to_3", "bottom"): "RH",
    ("3_to_6", "top"): "LH",
    ("3_to_6", "left"): "RH",
    ("6_to_9", "right"): "LH",
    ("6_to_9", "top"): "RH",
    ("9_to_12", "bottom"): "LH",
    ("9_to_12", "right"): "RH",
}

YOLO_CONFIDENCE = 0.25
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

mcp = FastMCP("floor-plan-door-detection")


def _map_derp(arc_type: str, opening_side: str) -> str:
    return DERP_MAP.get((arc_type, opening_side), "review")


def _create_gemini_client(api_key: str) -> tuple[genai.Client, httpx.Client]:
    http_client = httpx.Client(
        timeout=httpx.Timeout(
            connect=GEMINI_CONNECT_TIMEOUT_S,
            read=GEMINI_READ_TIMEOUT_S,
            write=GEMINI_WRITE_TIMEOUT_S,
            pool=GEMINI_POOL_TIMEOUT_S,
        ),
        follow_redirects=True,
        transport=httpx.HTTPTransport(retries=GEMINI_CONNECT_RETRIES),
    )
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(httpxClient=http_client),
    )
    return client, http_client


def _build_door_analysis_content(door_inputs: list[dict]) -> types.Content:
    parts = [types.Part.from_text(text=DOOR_DERP_PROMPT)]
    for door_input in door_inputs:
        parts.append(types.Part.from_text(text=f"door_id: {door_input['door_id']}"))
        parts.append(
            types.Part.from_bytes(
                data=door_input["image_bytes"],
                mime_type=door_input["mime_type"],
            )
        )
    return types.Content(role="user", parts=parts)


def _analyse_doors(client: genai.Client, door_inputs: list[dict]) -> list[dict]:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_door_analysis_content(door_inputs),
        config=types.GenerateContentConfig(
            temperature=0,
            responseMimeType="application/json",
            responseSchema=DOOR_ANALYSIS_RESPONSE_SCHEMA,
        ),
    )

    parsed = response.parsed
    if isinstance(parsed, dict):
        analysis = parsed
    elif hasattr(parsed, "model_dump"):
        analysis = parsed.model_dump()
    else:
        response_text = (response.text or "").strip()
        if response_text:
            analysis = json.loads(response_text)
        else:
            raise ValueError("Gemini returned no structured door analysis.")

    logger.info("Gemini raw response: %s", json.dumps(analysis, indent=2))

    doors = analysis.get("doors")
    if not isinstance(doors, list):
        raise ValueError("Gemini response is missing a doors list.")
    if len(doors) != len(door_inputs):
        raise ValueError(
            f"Gemini returned {len(doors)} doors for {len(door_inputs)} inputs."
        )

    validated = []
    for door, expected in zip(doors, door_inputs):
        if door.get("door_id") != expected["door_id"]:
            raise ValueError(
                f"Gemini returned door_id {door.get('door_id')!r}, "
                f"expected {expected['door_id']!r}."
            )
        validated.append(
            {
                "door_id": door["door_id"],
                "opening_side": door.get("opening_side", "unknown"),
                "arc_type": door.get("arc_type", "unknown"),
            }
        )
    return validated


def _build_review_results(raw_bboxes: list[list[int]]) -> list[dict]:
    return [
        {
            "door_id": f"Door_{i}.png",
            "bbox": bbox,
            "derp": "review",
        }
        for i, bbox in enumerate(raw_bboxes)
    ]


@mcp.tool()
def ping() -> str:
    """Simple health check to verify the MCP server is reachable."""
    logger.info("ping called")
    return "pong"


@mcp.tool()
def detect_doors(image_path: str) -> list[dict]:
    """
    Detect doors in a floor plan image and return their bounding boxes and DERP values. Use to find derp values for a door.

    Args:
        image_path: Absolute path to the floor plan image (PNG or JPEG) on the shared volume.
            Example: /shared/floorplan.png

    Returns:
        List of door objects, each with:
          - door_id: identifier string (e.g. "Door_0.png")
          - bbox: [x1, y1, x2, y2] pixel coordinates in the original image
          - derp: "LH" (left-hand), "RH" (right-hand), or "review"
    """
    logger.info("detect_doors called: %s", image_path)
    try:
        return _detect_doors_impl(image_path)
    except Exception:
        logger.exception("detect_doors: unhandled error for %s", image_path)
        raise


def _wait_for_file_ready(path: str, timeout: float = 30.0, interval: float = 0.5) -> None:
    """Poll until the file size stops changing, indicating the write is complete."""
    deadline = time.monotonic() + timeout
    last_size = -1
    while time.monotonic() < deadline:
        size = os.path.getsize(path)
        if size > 0 and size == last_size:
            return
        last_size = size
        time.sleep(interval)
    raise TimeoutError(f"File did not stabilise within {timeout}s: {path}")


def _detect_doors_impl(image_path: str) -> list[dict]:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    if not os.path.isfile(image_path):
        logger.error("detect_doors: file not found: %s", image_path)
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Wait for the file to finish writing by polling until size stabilizes
    _wait_for_file_ready(image_path)

    image = PIL.Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    logger.info("Image loaded: %dx%d", img_w, img_h)

    # Run YOLO detection, filter to Door class only
    model = YOLO(MODEL_PATH)
    results = model.predict(image, conf=YOLO_CONFIDENCE, verbose=False)
    door_boxes = [
        box for box in results[0].boxes if model.names[int(box.cls)] == "Door"
    ]
    logger.info("YOLO detected %d door(s)", len(door_boxes))

    if not door_boxes:
        logger.info("No doors detected, returning empty list")
        return []

    # Crop each door with 30% padding (matching app.py behaviour)
    door_inputs = []
    raw_bboxes = []
    for i, box in enumerate(door_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_bboxes.append([x1, y1, x2, y2])
        logger.info("Door_%d bbox: [%d, %d, %d, %d]", i, x1, y1, x2, y2)

        pad_x = int((x2 - x1) * 0.4)
        pad_y = int((y2 - y1) * 0.4)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(img_w, x2 + pad_x)
        cy2 = min(img_h, y2 + pad_y)

        crop = image.crop((cx1, cy1, cx2, cy2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")

        door_id = f"Door_{i}.png"
        door_inputs.append(
            {
                "door_id": door_id,
                "image_bytes": buf.getvalue(),
                "mime_type": "image/png",
            }
        )

    # Send door crops to Gemini in batches to stay within the 120s MCP timeout
    logger.info(
        "Sending %d door crop(s) to Gemini in batches of %d",
        len(door_inputs),
        GEMINI_DOOR_BATCH_SIZE,
    )
    http_client = None
    try:
        gemini_client, http_client = _create_gemini_client(gemini_api_key)
        analysed = []
        for batch_start in range(0, len(door_inputs), GEMINI_DOOR_BATCH_SIZE):
            batch = door_inputs[batch_start : batch_start + GEMINI_DOOR_BATCH_SIZE]
            logger.info(
                "Gemini batch %d-%d of %d",
                batch_start,
                batch_start + len(batch) - 1,
                len(door_inputs),
            )
            analysed.extend(_analyse_doors(gemini_client, batch))
    except Exception:
        logger.exception(
            "detect_doors: Gemini door analysis failed for %s; returning YOLO detections with review DERP",
            image_path,
        )
        return _build_review_results(raw_bboxes)
    finally:
        if http_client is not None:
            http_client.close()
    logger.info("Gemini analysis complete")

    # Map (arc_type, opening_side) -> DERP and build result list
    results_out = [
        {
            "door_id": door["door_id"],
            "bbox": raw_bboxes[i],
            "derp": _map_derp(door["arc_type"], door["opening_side"]),
        }
        for i, door in enumerate(analysed)
    ]
    for r in results_out:
        logger.info("%s -> derp=%s bbox=%s", r["door_id"], r["derp"], r["bbox"])
    return results_out


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info("MCP server starting on port %d", port)
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
