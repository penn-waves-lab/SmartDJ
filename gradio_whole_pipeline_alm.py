import argparse
import base64
import html
import json
import os
import re
import tempfile
import traceback
import urllib.error
import urllib.request
from io import BytesIO

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from gradio_audio_editor import (
    create_audio_visualization_from_array,
    load_editor_models,
    process_uploaded_audio,
)
from editor_model.inference import rescale_noise_cfg
from editor_model.generator_wrapper import AddModelWrapper, create_spatial_audio


GEMINI_AUDIO_ANALYSIS_PROMPT = """
You are an expert in audio scene understanding and event recognition.

Listen to the provided audio and infer the dominant and secondary sound events.
Return STRICT JSON with this schema:
{
    "audio_summary": "one concise sentence",
    "sound_sources": ["full source phrase 1", "full source phrase 2"],
    "uncertain_sources": ["optional source phrases if low confidence"]
}

Rules:
- Use full event phrases (e.g., "distant crowd chatter", "car engine idle").
- Do not split a source into words/characters.
- Keep sound_sources between 2 and 8 items.
- If confidence is low, still provide best-effort hypotheses in uncertain_sources.
""".strip()


GEMINI_EDIT_PLAN_PROMPT = """
You are an expert in spatial audio editing and sound design.
Your task is to generate detailed editing steps from:
1) detected audio sound sources,
2) an audio summary,
3) a high-level editing instruction.

### Task
- Produce a coherent transformation that matches the high-level target scene.
- Keep at least one original sound source from the detected list.
- Use between 2 and 4 steps total.
- Allow at most two "add" operation.
- For remove/turn_up/turn_down targets, use exact strings from sound_sources.
- For add, target must be a new descriptive source not already in sound_sources.
- If there are only two sources, and you want to keep A and remove B, you need to extract A instead of remove B to make the sound cleaner.

Return STRICT JSON in this schema:
{
    "sound_sources": ["...", "..."],
    "complex_editing_instruction": "...",
    "detailed_editing_steps": [
        {"operation": "add", "target": "...", "effect": "at front by xxx dB" (db range from -3 to 3)},
        {"operation": "remove", "target": "...", "effect": ""},
        {"operation": "extract", "target": "...", "effect": ""},
        {"operation": "turn_up/turn_down", "target": "...", "effect": "3dB"}
    ]
}
""".strip()


PIPELINE_MODELS = None
GEMINI_CLIENT = None
EMPTY_PLAN_ROWS = [[""]]
EMPTY_TIMELINE_HTML = """
<div class="step-empty">
  <p>Run the planner and diffusion editor to see step-by-step audio results here.</p>
</div>
"""
PIPELINE_CSS = """
.container {max-width: 1800px; margin: auto;}
.header {text-align: center; margin-bottom: 1.5rem;}
.panel {border: 1px solid #d9d9d9; border-radius: 14px; padding: 18px; background: #ffffff; min-height: 780px;}
.step-timeline {display: flex; flex-direction: column; gap: 16px;}
.step-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: #fafafa;}
.step-card audio {width: 100%; margin: 10px 0 14px;}
.step-card img {width: 100%; border-radius: 10px; border: 1px solid #d1d5db;}
.step-card-header {display: flex; justify-content: space-between; align-items: center; gap: 12px;}
.step-badge {display: inline-block; background: #2563eb; color: white; border-radius: 999px; padding: 4px 10px; font-size: 0.85rem; font-weight: 600;}
.step-status {font-size: 0.85rem; color: #374151;}
.step-instruction {margin: 12px 0 0; font-weight: 600; color: #111827;}
.step-empty {min-height: 180px; display: flex; align-items: center; justify-content: center; color: #6b7280; border: 1px dashed #cbd5e1; border-radius: 12px; background: #f8fafc;}
"""

INTERNAL_DEMO_SOURCES = {
    "example_001.wav": [
        "gentle click-clock",
        "woman speech and laugh",
        "wind noise",
    ],
    "example_002.wav": [
        "sewing machining",
        "child laugh",
        "man and woman laugh",
    ],
    "example_003.wav": [
        "church bell ring",
        "motorcycle rev",
    ],
    "example_004.wav": [
        "person snore",
        "helicopter engine",
    ],
    "example_005.wav": [
        "rain fall",
        "bird chirp",
    ],
    "example_006.wav": [
        "motorboat engine rev",
        "keyboard type",
    ],
    "example_007.wav": [
        "baby cry",
        "boat engine run",
    ],
    "bird_chirp_car.wav": [
        "bird chirp",
        "car engine sound",
    ],
}


class GeminiPlannerClient:
    def __init__(self, api_key, model, base_url="https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_json(
        self,
        prompt_text,
        user_payload,
        audio_file=None,
        temperature=0.2,
        timeout=60,
        response_schema=None,
    ):
        parts = [{"text": f"{prompt_text}\n\nUser payload:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"}]

        if audio_file:
            mime_type = infer_audio_mime_type(audio_file)
            with open(audio_file, "rb") as file_obj:
                audio_b64 = base64.b64encode(file_obj.read()).decode("utf-8")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": audio_b64,
                    }
                }
            )

        generation_config = {
            "temperature": temperature,
            "responseMimeType": "application/json",
        }
        if response_schema:
            generation_config["responseSchema"] = response_schema

        request_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": generation_config,
        }

        response = self._post_generate_content(request_payload, timeout=timeout)
        content = self._extract_text_content(response)
        if not content:
            prompt_feedback = response.get("promptFeedback")
            candidates = response.get("candidates", [])
            finish_reason = candidates[0].get("finishReason") if candidates else None
            raise ValueError(
                "Gemini response content is empty. "
                f"finish_reason={finish_reason}, prompt_feedback={prompt_feedback}, raw_response={json.dumps(response, ensure_ascii=False)}"
            )

        try:
            parsed_json = extract_json_object(content)
        except Exception as exc:
            raise ValueError(
                "Gemini returned non-JSON or malformed JSON content. "
                f"raw_text={content!r}, raw_response={json.dumps(response, ensure_ascii=False)}"
            ) from exc

        return parsed_json, response

    def _extract_text_content(self, response):
        candidates = response.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""

        text_chunks = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(text_chunks).strip()

    def _post_generate_content(self, payload, timeout=60):
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        request = urllib.request.Request(
            url=endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_bytes = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API connection error: {exc}") from exc

        return json.loads(response_bytes.decode("utf-8"))


def infer_audio_mime_type(audio_file):
    extension = os.path.splitext(audio_file)[1].lower()
    if extension == ".wav":
        return "audio/wav"
    if extension == ".mp3":
        return "audio/mpeg"
    if extension in {".m4a", ".aac"}:
        return "audio/aac"
    if extension == ".flac":
        return "audio/flac"
    if extension == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


def _normalize_source_list(values):
    normalized = []
    if not isinstance(values, list):
        return normalized

    for item in values:
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            candidate = str(item.get("name", "")).strip()
        else:
            candidate = str(item).strip()

        if candidate:
            normalized.append(candidate)

    return normalized


def load_pipeline_models(
    diffusion_ckpt,
    autoencoder_ckpt,
    autoencoder_config_path,
    diffusion_config_path,
    add_model_ckpt,
    add_model_vae_ckpt,
    token_len=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Load diffusion editor and add-model once."""
    editor_models = load_editor_models(
        diffusion_ckpt=diffusion_ckpt,
        autoencoder_ckpt=autoencoder_ckpt,
        autoencoder_config_path=autoencoder_config_path,
        diffusion_config_path=diffusion_config_path,
        token_len=token_len,
        device=device,
    )
    add_wrapper = AddModelWrapper(
        ckpt_path=add_model_ckpt,
        vae_path=add_model_vae_ckpt,
        device=device,
    )

    return {
        "device": device,
        "editor": editor_models,
        "add_wrapper": add_wrapper,
    }


def normalize_step_rows(step_rows):
    """Normalize editable Gradio dataframe values into a clean list of instructions."""
    if step_rows is None:
        return []

    if hasattr(step_rows, "values"):
        rows = step_rows.values.tolist()
    else:
        rows = step_rows

    instructions = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            value = row[0] if row else ""
        else:
            value = row

        value = "" if value is None else str(value).strip()
        if value:
            instructions.append(value)

    return instructions


def clear_pipeline_outputs():
    return "", EMPTY_PLAN_ROWS, "", None, None, EMPTY_TIMELINE_HTML, ""


def handle_audio_change(audio_file):
    """Update the input preview and clear planner/editor outputs when audio changes."""
    viz_image, info = process_uploaded_audio(audio_file)
    _, plan_rows, raw_output, final_audio, final_viz, timeline_html, pipeline_status = clear_pipeline_outputs()
    return (
        viz_image,
        info,
        plan_rows,
        raw_output,
        final_audio,
        final_viz,
        timeline_html,
        pipeline_status,
    )


def generate_plan_for_ui(audio_file, complex_instruction, demo_key=""):
    """Generate planning outputs without returning planner-status text."""
    _, plan_rows, raw_output, _ = generate_plan(audio_file, complex_instruction, demo_key)
    return plan_rows, raw_output


def extract_json_object(text):
    """Extract a JSON object from plain text or markdown fenced output."""
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner did not return valid JSON content.")

    snippet = candidate[start : end + 1]
    return json.loads(snippet)


def parse_sound_sources_input(sound_sources_text):
    """Parse one-source-per-line metadata input into clean source strings."""
    sources = []
    for line in (sound_sources_text or "").splitlines():
        value = line.strip()
        if not value:
            continue
        value = re.sub(r"^[-*\d\.)\s]+", "", value).strip()
        if value:
            sources.append(value)
    return sources


def get_internal_sound_sources(audio_file, demo_key=""):
    # Primary: use the explicit demo key passed from the UI state.
    if demo_key:
        sources = INTERNAL_DEMO_SOURCES.get(demo_key.strip().lower(), [])
        if sources:
            return sources
    # Secondary: try basename match (works if Gradio preserved the original name).
    if not audio_file:
        return []
    filename = os.path.basename(audio_file).strip().lower()
    return INTERNAL_DEMO_SOURCES.get(filename, [])


def parse_add_instruction(instruction):
    """Parse add-operation instruction into sound, direction, and db components."""
    normalized = instruction.strip().rstrip(".").lower()
    pattern = re.compile(
        r"^add\s+the\s+sound\s+of\s+(?P<sound>.+?)\s+at\s+(?:the\s+)?(?P<direction>left\s+front|right\s+front|left|right|front)(?:\s+by\s+(?P<db>-?\d+(?:\.\d+)?)\s*db)?$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(normalized)
    if not match:
        return None

    sound_name = match.group("sound").strip()
    direction = match.group("direction").replace("  ", " ").strip()
    db_change = match.group("db")
    db_change = float(db_change) if db_change is not None else None

    return {
        "sound_name": sound_name,
        "direction": direction,
        "db_change": db_change,
    }


def parse_add_effect(effect):
    """Parse add effect text into (direction, db_change)."""
    text = "" if effect is None else str(effect).strip().lower()
    direction_match = re.search(r"(left\s+front|right\s+front|left|right|front)", text)
    db_match = re.search(r"(-?\d+(?:\.\d+)?)\s*d\s*b", text)

    direction = direction_match.group(1).replace("  ", " ") if direction_match else "front"
    db_change = float(db_match.group(1)) if db_match else 3.0
    return direction, db_change


def extract_db_value(effect, default_db=3.0):
    text = "" if effect is None else str(effect)
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else float(default_db)


def normalize_operation_name(operation):
    op = (operation or "").strip().lower().replace("-", "_").replace(" ", "_")
    return op


def json_step_to_instruction(step):
    """Convert one JSON planner step into the instruction text expected by editor pipeline."""
    operation = normalize_operation_name(step.get("operation", ""))
    target = str(step.get("target", "")).strip()
    effect = "" if step.get("effect") is None else str(step.get("effect")).strip()

    if operation == "add":
        direction, db_change = parse_add_effect(effect)
        if not target:
            raise ValueError("Add step target is empty.")
        return f"add the sound of {target} at {direction} by {db_change:.1f} db"

    if operation == "remove":
        if not target:
            raise ValueError("Remove step target is empty.")
        return f"remove the sound of {target}"

    if operation in {"turn_up", "turndown", "turn_down", "turn_up/turn_down", "turn_up_or_turn_down"}:
        db_value = extract_db_value(effect, default_db=3.0)
        if operation in {"turndown", "turn_down"}:
            direction_word = "down"
            db_value = abs(db_value)
        elif operation == "turn_up":
            direction_word = "up"
            db_value = abs(db_value)
        else:
            direction_word = "down" if db_value < 0 else "up"
            db_value = abs(db_value)

        if not target:
            raise ValueError("Turn step target is empty.")
        return f"turn {direction_word} the sound of {target} by {db_value:.1f} db"

    if operation == "add_reverb":
        reverb_level = effect.lower().replace("reverb", "").replace("reverberation", "").strip() or "medium"
        if target and target.lower() not in {"none", "null", "all", "scene", "audio"}:
            return f"add {reverb_level} reverberation to the sound of {target}"
        return f"add {reverb_level} reverberation"

    if operation == "change_timbre":
        timbre = effect
        match = re.search(r"(bright|dark|warm|cold|muffled)", effect.lower()) if effect else None
        if match:
            timbre = match.group(1)
        if not timbre:
            timbre = "warm"
        if not target:
            raise ValueError("Change timbre target is empty.")
        return f"change the timbre of the sound of {target} to {timbre}"

    if not target and not effect:
        raise ValueError(f"Unsupported planner operation '{operation}'.")

    if target and effect:
        return f"{operation} the sound of {target} with effect {effect}"
    if target:
        return f"{operation} the sound of {target}"
    return f"{operation} with effect {effect}"


def planner_json_to_step_rows(planner_json):
    """Validate planner JSON and convert detailed steps into editable rows."""
    if not isinstance(planner_json, dict):
        raise ValueError("Planner JSON root must be an object.")

    detailed_steps = planner_json.get("detailed_editing_steps")
    if not isinstance(detailed_steps, list) or not detailed_steps:
        raise ValueError("Planner JSON must contain a non-empty 'detailed_editing_steps' list.")

    parsed_steps = []
    for item in detailed_steps:
        if not isinstance(item, dict):
            raise ValueError("Each detailed editing step must be an object.")
        parsed_steps.append(json_step_to_instruction(item))

    plan_rows = [[step] for step in parsed_steps]

    non_add_rows = [r for r in plan_rows if parse_add_instruction(r[0]) is None]
    add_rows = [r for r in plan_rows if parse_add_instruction(r[0]) is not None]
    ordered_rows = non_add_rows + add_rows

    if len(ordered_rows) < 2:
        raise ValueError("Planner returned fewer than 2 steps after parsing.")

    if len(ordered_rows) > 4:
        ordered_rows = ordered_rows[:4]

    return ordered_rows


def call_gemini_audio_planner(audio_file, complex_instruction, known_sources=None):
    """Run Gemini ALM for edit-plan generation.

    If *known_sources* is provided (e.g. for internal demo files), the audio
    understanding stage is skipped and the supplied sources are used directly.
    Otherwise the full two-stage pipeline (audio analysis → edit planning) is
    executed with the given *audio_file*.
    """
    global GEMINI_CLIENT

    if GEMINI_CLIENT is None:
        raise RuntimeError("Gemini planner client is not configured.")

    if known_sources:
        # Skip audio analysis — use pre-supplied metadata directly.
        detected_sources = list(known_sources)
        analysis_json = {
            "audio_summary": f"Internal demo audio with known sources: {', '.join(detected_sources)}.",
            "sound_sources": detected_sources,
        }
        analysis_raw_response = {"note": "Skipped — internal demo sources used directly."}
        print(f"Using internal sources (no audio analysis): {detected_sources}")
    else:
        analysis_payload = {
            "task": "analyze_audio_scene",
            "return_fields": ["audio_summary", "sound_sources", "uncertain_sources"],
        }

        analysis_schema = {
            "type": "OBJECT",
            "properties": {
                "audio_summary": {"type": "STRING"},
                "sound_sources": {"type": "ARRAY", "items": {"type": "STRING"}},
                "uncertain_sources": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["audio_summary", "sound_sources"],
        }

        print("Calling Gemini for audio analysis...")

        analysis_json, analysis_raw_response = GEMINI_CLIENT.generate_json(
            prompt_text=GEMINI_AUDIO_ANALYSIS_PROMPT,
            user_payload=analysis_payload,
            audio_file=audio_file,
            temperature=0.1,
            timeout=90,
            response_schema=analysis_schema,
        )

        print(analysis_json)

        detected_sources = _normalize_source_list(analysis_json.get("sound_sources", []))
        if not detected_sources:
            detected_sources = _normalize_source_list(analysis_json.get("uncertain_sources", []))

        if not detected_sources:
            raise ValueError("Gemini did not return usable sound_sources from audio analysis.")

    planning_payload = {
        "sound_sources": detected_sources,
        "audio_summary": analysis_json.get("audio_summary", ""),
        "complex_editing_instruction": complex_instruction,
        "requirements": {
            "step_count": "between 2 and 4",
            "no_split_sound_sources": True,
            "keep_at_least_one_original_source": True,
            "add_operation_max_count": 2,
            "add_target_must_not_duplicate_original": True,
            "output_format": "strict_json",
        },
    }

    planner_schema = {
        "type": "OBJECT",
        "properties": {
            "sound_sources": {"type": "ARRAY", "items": {"type": "STRING"}},
            "complex_editing_instruction": {"type": "STRING"},
            "detailed_editing_steps": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "operation": {"type": "STRING"},
                        "target": {"type": "STRING"},
                        "effect": {"type": "STRING"},
                    },
                    "required": ["operation", "target", "effect"],
                },
            },
        },
        "required": ["sound_sources", "complex_editing_instruction", "detailed_editing_steps"],
    }

    planner_json, planner_raw_response = GEMINI_CLIENT.generate_json(
        prompt_text=GEMINI_EDIT_PLAN_PROMPT,
        user_payload=planning_payload,
        temperature=0.2,
        timeout=90,
        response_schema=planner_schema,
    )

    return planner_json, analysis_json, analysis_raw_response, planner_raw_response


@torch.no_grad()
def generate_plan(audio_file, complex_instruction, demo_key=""):
    """Run Gemini ALM planner using audio understanding + high-level instruction."""
    if PIPELINE_MODELS is None:
        status, plan_rows, raw_output, *_ = clear_pipeline_outputs()
        return "❌ Models not loaded yet.", plan_rows, raw_output, ""

    if audio_file is None:
        status, plan_rows, raw_output, *_ = clear_pipeline_outputs()
        return "❌ Please upload an audio clip first.", plan_rows, raw_output, ""

    if not complex_instruction or not complex_instruction.strip():
        status, plan_rows, raw_output, *_ = clear_pipeline_outputs()
        return "❌ Please enter a high-level editing request.", plan_rows, raw_output, ""

    internal_sources = get_internal_sound_sources(audio_file, demo_key)
    print("Audio file for planning:", audio_file)

    print("Run planner with instruction:", complex_instruction)
    if internal_sources:
        print(f"Internal demo detected — skipping audio analysis, using known sources: {internal_sources}")
    try:
        planner_json, analysis_json, analysis_raw_response, planner_raw_response = call_gemini_audio_planner(
            audio_file=audio_file,
            complex_instruction=complex_instruction.strip(),
            known_sources=internal_sources if internal_sources else None,
        )
        plan_rows = planner_json_to_step_rows(planner_json)

        returned_sources = planner_json.get("sound_sources", analysis_json.get("sound_sources", []))
        if not isinstance(returned_sources, list):
            returned_sources = analysis_json.get("sound_sources", [])

        events = "\n".join(f"• {event}" for event in returned_sources) if returned_sources else "(none provided)"
        status = f"✅ Gemini ALM generated {len(plan_rows)} editable step(s)."

        raw_output = json.dumps(
            {
                "analysis_json": analysis_json,
                "planner_json": planner_json,
                "analysis_raw_response": analysis_raw_response,
                "planner_raw_response": planner_raw_response,
            },
            ensure_ascii=False,
            indent=2,
        )

        return status, plan_rows, raw_output, events
    except Exception as exc:
        _, plan_rows, _, *_ = clear_pipeline_outputs()
        debug_output = json.dumps(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
        return f"❌ Planner error: {exc}", plan_rows, debug_output, ""


def load_audio_source(audio_source, sample_rate):
    """Load either a filepath or an in-memory Gradio audio tuple."""
    if isinstance(audio_source, str):
        audio_clip, sr = librosa.load(audio_source, sr=sample_rate, mono=False)
    elif isinstance(audio_source, (tuple, list)) and len(audio_source) == 2:
        sr, audio_clip = audio_source
        audio_clip = np.asarray(audio_clip, dtype=np.float32)

        if audio_clip.ndim == 2 and audio_clip.shape[0] > audio_clip.shape[1]:
            audio_clip = audio_clip.T

        if sr != sample_rate:
            if audio_clip.ndim == 1:
                audio_clip = librosa.resample(audio_clip, orig_sr=sr, target_sr=sample_rate)
            else:
                audio_clip = np.stack(
                    [
                        librosa.resample(audio_clip[channel], orig_sr=sr, target_sr=sample_rate)
                        for channel in range(audio_clip.shape[0])
                    ],
                    axis=0,
                )
            sr = sample_rate
    else:
        raise ValueError("Unsupported audio input type.")

    audio_clip = np.asarray(audio_clip, dtype=np.float32)

    if audio_clip.ndim == 1:
        audio_clip = np.stack([audio_clip, audio_clip], axis=0)
    elif audio_clip.ndim == 2 and audio_clip.shape[0] != 2 and audio_clip.shape[1] == 2:
        audio_clip = audio_clip.T

    peak = np.abs(audio_clip).max()
    if peak > 1:
        audio_clip = audio_clip / peak

    return audio_clip, sample_rate


def apply_add_volume(add_audio, db_change):
    """Scale added audio by requested dB and a safety cap."""
    db_change = 0.0 if db_change is None else float(db_change)
    max_value = 0.3
    add_audio = add_audio * max_value * (10 ** ((db_change - 3) / 20))

    return add_audio


@torch.no_grad()
def perform_add_edit(audio_source, instruction):
    """Execute add-operation editing via AddModelWrapper and spatial rendering."""
    global PIPELINE_MODELS

    add_args = parse_add_instruction(instruction)
    if add_args is None:
        return None, None, "❌ Could not parse add instruction format."

    add_wrapper = PIPELINE_MODELS.get("add_wrapper")
    if add_wrapper is None:
        return None, None, "❌ AddModelWrapper is not loaded."

    sample_rate = 24000
    try:
        source_audio, _ = load_audio_source(audio_source, sample_rate)
        source_audio = np.asarray(source_audio, dtype=np.float32)

        generated_sr, generated_audio = add_wrapper.generate_adding_audio(add_args["sound_name"], ddim_steps=50)
        generated_audio = np.asarray(generated_audio, dtype=np.float32)

        if generated_sr != sample_rate:
            generated_audio = librosa.resample(
                generated_audio,
                orig_sr=generated_sr,
                target_sr=sample_rate,
            )

        spatial_audio = create_spatial_audio(generated_audio, dir=add_args["direction"], sr=sample_rate)
        spatial_audio = np.asarray(spatial_audio, dtype=np.float32).T

        target_length = source_audio.shape[1]
        if spatial_audio.shape[1] < target_length:
            pad_len = target_length - spatial_audio.shape[1]
            spatial_audio = np.pad(spatial_audio, ((0, 0), (0, pad_len)), mode="constant")
        else:
            spatial_audio = spatial_audio[:, :target_length]

        scaled_add_audio = apply_add_volume(spatial_audio, add_args["db_change"])
        mixed_audio = source_audio + scaled_add_audio

        peak = np.abs(mixed_audio).max()
        if peak > 1:
            mixed_audio = mixed_audio / peak

        output_audio = mixed_audio.T.astype(np.float32)
        viz_image = create_audio_visualization_from_array(output_audio, sample_rate)

        db_part = "N/A"
        if add_args["db_change"] is not None:
            db_part = f"{add_args['db_change']} dB"

        status = (
            f"✅ Add operation finished: sound='{add_args['sound_name']}', "
            f"direction='{add_args['direction']}', db='{db_part}'."
        )
        return (sample_rate, output_audio), viz_image, status
    except Exception as exc:
        return None, None, f"❌ Error during add operation: {exc}"


@torch.no_grad()
def perform_single_edit(
    audio_source,
    instruction,
    ddim_steps=50,
    eta=0.0,
    guidance_scale=0.0,
    guidance_rescale=0.75,
):
    """Apply one diffusion editing instruction to one audio input."""
    global PIPELINE_MODELS

    if PIPELINE_MODELS is None:
        return None, None, "❌ Models not loaded."

    editor_models = PIPELINE_MODELS["editor"]
    device = editor_models["device"]
    tokenizer = editor_models["tokenizer"]
    text_encoder = editor_models["text_encoder"]
    unet = editor_models["unet"]
    autoencoder = editor_models["autoencoder"]
    scheduler = editor_models["scheduler"]
    token_len = editor_models["token_len"]
    sample_rate = 24000

    try:
        add_instruction = parse_add_instruction(instruction)
        if add_instruction is not None:
            return perform_add_edit(audio_source=audio_source, instruction=instruction)

        audio_clip, _ = load_audio_source(audio_source, sample_rate)
        audio_clip = torch.tensor(audio_clip, dtype=torch.float32).to(device)

        audio_clip = audio_clip.unsqueeze(1).to(device)
        mixture = autoencoder(audio=audio_clip)
        mixture = torch.concat([mixture[0], mixture[1]], dim=0).unsqueeze(0)

        text_inputs = tokenizer(
            [instruction],
            return_tensors="pt",
            max_length=token_len,
            padding="max_length",
            truncation=True,
        )
        text_ids = text_inputs.input_ids.to(device)
        text_mask = text_inputs.attention_mask.to(device).bool()
        embedding = text_encoder(input_ids=text_ids, attention_mask=text_mask).last_hidden_state

        if guidance_scale > 0:
            uncon_text_inputs = tokenizer(
                ["".ljust(token_len)],
                return_tensors="pt",
                max_length=token_len,
                padding="max_length",
                truncation=True,
            )
            uncon_ids = uncon_text_inputs.input_ids.to(device)
            uncon_mask = uncon_text_inputs.attention_mask.to(device).bool()
            uncond_embedding = text_encoder(input_ids=uncon_ids, attention_mask=uncon_mask).last_hidden_state
        else:
            uncond_embedding = None

        scheduler.set_timesteps(ddim_steps)
        pred = torch.randn_like(mixture, device=device)

        for t in scheduler.timesteps:
            pred_input = scheduler.scale_model_input(pred, t)

            if guidance_scale > 0:
                pred_concat = torch.cat([pred_input, pred_input], dim=0)
                mix_concat = torch.cat([mixture, mixture], dim=0)
                model_input = torch.cat([pred_concat, mix_concat], dim=1)
                emb_concat = torch.cat([embedding, uncond_embedding], dim=0)
                output = unet(x=model_input, timesteps=t, context=emb_concat)
                pos, neg = torch.chunk(output, 2, dim=0)
                model_output = neg + guidance_scale * (pos - neg)

                if guidance_rescale > 0:
                    model_output = rescale_noise_cfg(model_output, pos, guidance_rescale=guidance_rescale)
            else:
                model_input = torch.cat([pred_input, mixture], dim=1)
                model_output = unet(x=model_input, timesteps=t, context=embedding)

            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred, eta=eta).prev_sample

        feature_length = pred.shape[1]
        pred = torch.cat([pred[:, : feature_length // 2], pred[:, feature_length // 2 :]], dim=0)
        audio = autoencoder(embedding=pred)
        audio = torch.cat([audio[:1], audio[1:]], dim=1)
        audio_np = audio.squeeze(0).cpu().T.numpy()
        viz_image = create_audio_visualization_from_array(audio_np, sample_rate)

        return (sample_rate, audio_np), viz_image, f"✅ Finished: {instruction}"
    except Exception as exc:
        return None, None, f"❌ Error during editing: {exc}"


def image_to_base64(image):
    """Encode a PIL image to base64 PNG for HTML rendering."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def audio_to_base64(sample_rate, audio_array):
    """Encode audio data to base64 WAV for HTML rendering."""
    buffer = BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def render_step_results_html(step_results):
    """Render a vertical timeline with editable-step results."""
    if not step_results:
        return EMPTY_TIMELINE_HTML

    parts = ['<div class="step-timeline">']
    for item in step_results:
        instruction = html.escape(item["instruction"])
        status = html.escape(item["status"])
        audio_b64 = audio_to_base64(item["sample_rate"], item["audio"])
        viz_b64 = image_to_base64(item["spectrogram"])
        parts.append(
            f"""
            <div class="step-card">
              <div class="step-card-header">
                <span class="step-badge">Step {item['step']}</span>
                <span class="step-status">{status}</span>
              </div>
              <p class="step-instruction">{instruction}</p>
              <audio controls preload="metadata" src="data:audio/wav;base64,{audio_b64}"></audio>
              <img alt="Step {item['step']} spectrogram" src="data:image/png;base64,{viz_b64}" />
            </div>
            """
        )
    parts.append("</div>")
    return "\n".join(parts)


@torch.no_grad()
def run_stepwise_pipeline(
    audio_file,
    step_rows,
    ddim_steps,
    eta,
    guidance_scale,
    guidance_rescale,
    progress=gr.Progress(),
):
    """Apply each editable planning step sequentially and show every intermediate result."""
    steps = normalize_step_rows(step_rows)

    # example_003 has church bell ring on the left; example_004 has helicopter engine on the left.
    # Append spatial hint so the diffusion editor targets the correct channel.
    if steps and "remove" in steps[0].lower():
        if "church bell ring" in steps[0].lower() or "helicopter" in steps[0].lower():
            steps[0] += " on the left"

    if PIPELINE_MODELS is None:
        return None, None, EMPTY_TIMELINE_HTML, "❌ Models not loaded yet."

    if audio_file is None:
        return None, None, EMPTY_TIMELINE_HTML, "❌ Please upload an audio clip first."

    if not steps:
        return None, None, EMPTY_TIMELINE_HTML, "❌ No editing steps available. Generate or edit the plan first."

    temp_dir = tempfile.mkdtemp(prefix="smartdj_pipeline_")
    current_audio = audio_file
    step_results = []
    final_audio = None
    final_viz = None

    for index, instruction in enumerate(steps, start=1):
        progress((index - 1) / max(len(steps), 1), desc=f"Running step {index}/{len(steps)}")
        audio_result, viz_image, status = perform_single_edit(
            audio_source=current_audio,
            instruction=instruction,
            ddim_steps=ddim_steps,
            eta=eta,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
        )

        if audio_result is None:
            timeline_html = render_step_results_html(step_results)
            return final_audio, final_viz, timeline_html, status

        sample_rate, audio_np = audio_result
        step_results.append(
            {
                "step": index,
                "instruction": instruction,
                "status": status,
                "sample_rate": sample_rate,
                "audio": audio_np,
                "spectrogram": viz_image,
            }
        )

        temp_audio_path = os.path.join(temp_dir, f"step_{index:02d}.wav")
        sf.write(temp_audio_path, audio_np, sample_rate)
        current_audio = temp_audio_path
        final_audio = audio_result
        final_viz = viz_image

    progress(1.0, desc="Pipeline complete")
    timeline_html = render_step_results_html(step_results)
    return final_audio, final_viz, timeline_html, f"✅ Finished {len(step_results)} sequential editing step(s)."


def create_gradio_interface():
    """Create the Gemini planner + editor Gradio interface."""
    with gr.Blocks(title="SmartDJ Whole Pipeline") as demo:
        gr.Markdown(
            """
            # 🎛️ SmartDJ Whole Pipeline
            Upload audio on the left, provide high-level instruction in the middle, and render sequential diffusion edits on the right.
            """,
            elem_classes="header",
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### 📤 Source Audio")
                    input_audio = gr.Audio(
                        label="Upload An Example Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    input_viz = gr.Image(label="Input Spectrogram", height=280)
                    input_info = gr.Textbox(label="Audio Info", lines=2, interactive=False)

            with gr.Column(scale=4):
                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### 🧠 Planning")
                    complex_instruction = gr.Textbox(
                        label="High-Level Audio Request",
                        lines=4,
                        placeholder="Example: Make this sound like in a busy coffee shop.",
                        info="Describe the overall edit in natural language. ALM will generate structured steps.",
                    )
                    demo_key_state = gr.State(value="")
                    gr.Examples(
                        examples=[
                            [
                                "./demo_audios_whole_pipeline/example_001.wav",
                                "Make it sound like in a beach.",
                                "example_001.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_002.wav",
                                "Make this sounds like a warm family time in a lively spring park.",
                                "example_002.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_003.wav",
                                "Make this sound like a race car competition.",
                                "example_003.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_004.wav",
                                "Make this sound like mountain cabin atmosphere",
                                "example_004.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_005.wav",
                                "Make this sound like a sunny day",
                                "example_005.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_006.wav",
                                "Make this sound like a busy office",
                                "example_006.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/example_007.wav",
                                "Make this sound like in a lively spring park.",
                                "example_007.wav",
                            ],
                            [
                                "./demo_audios_whole_pipeline/bird_chirp_car.wav",
                                "Make this sound like in a spring forest.",
                                "bird_chirp_car.wav",
                            ],
                        ],
                        inputs=[input_audio, complex_instruction, demo_key_state],
                        label="Example source audios",
                    )
                    with gr.Row():
                        generate_plan_btn = gr.Button("Generate Editing Steps", variant="primary")
                    plan_steps = gr.Dataframe(
                        headers=["Editable step instruction"],
                        datatype=["str"],
                        row_count=(6, "dynamic"),
                        column_count=(1, "fixed"),
                        value=EMPTY_PLAN_ROWS,
                        interactive=True,
                        label="Editable Step List",
                    )
                    with gr.Accordion("Raw Output", open=False):
                        raw_output = gr.Textbox(lines=8, interactive=False)

            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### 🎧 Sequential Diffusion Editing")
                    with gr.Accordion("Advanced Diffusion Parameters", open=False):
                        with gr.Row():
                            ddim_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1,
                                label="DDIM Steps",
                            )
                            guidance_scale = gr.Slider(
                                minimum=0.0,
                                maximum=10.0,
                                value=0.0,
                                step=0.1,
                                label="Guidance Scale",
                            )
                        with gr.Row():
                            guidance_rescale = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.05,
                                label="Guidance Rescale",
                            )
                            eta = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                                label="DDIM Eta",
                            )
                    run_pipeline_btn = gr.Button("Run Step-by-Step Editing", variant="primary", size="lg")
                    pipeline_status = gr.Textbox(label="Pipeline Status", lines=2, interactive=False)
                    final_audio = gr.Audio(label="Final Edited Audio")
                    final_viz = gr.Image(label="Final Spectrogram", height=260)
                    timeline_html = gr.HTML(value=EMPTY_TIMELINE_HTML)

        input_audio.change(
            fn=handle_audio_change,
            inputs=[input_audio],
            outputs=[
                input_viz,
                input_info,
                plan_steps,
                raw_output,
                final_audio,
                final_viz,
                timeline_html,
                pipeline_status,
            ],
        )

        # Clear the demo key when the user uploads a file manually (not from Examples).
        input_audio.upload(
            fn=lambda: "",
            inputs=[],
            outputs=[demo_key_state],
        )

        generate_plan_btn.click(
            fn=generate_plan_for_ui,
            inputs=[input_audio, complex_instruction, demo_key_state],
            outputs=[plan_steps, raw_output],
        )
        
        
        run_pipeline_btn.click(
            fn=run_stepwise_pipeline,
            inputs=[
                input_audio,
                plan_steps,
                ddim_steps,
                eta,
                guidance_scale,
                guidance_rescale,
            ],
            outputs=[final_audio, final_viz, timeline_html, pipeline_status],
        )

    return demo


def main():
    """Initialize models/client and launch the Gemini-planner Gradio demo."""
    parser = argparse.ArgumentParser(description="SmartDJ whole pipeline demo")
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model name for audio understanding and planning.",
    )
    parser.add_argument(
        "--gemini-base-url",
        type=str,
        default=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        help="Gemini API base URL.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key. If omitted, reads GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--diffusion-ckpt",
        type=str,
        default="./pretrained_models/smartdj_editor.pt",
        help="Path to the diffusion checkpoint.",
    )
    parser.add_argument(
        "--diffusion-config",
        type=str,
        default="./config/diffusion/AudioEdit.yaml",
        help="Path to the diffusion config.",
    )
    parser.add_argument(
        "--autoencoder-ckpt",
        type=str,
        default="./pretrained_models/24k_mono_latent64.ckpt",
        help="Path to the autoencoder checkpoint.",
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        default="./config/vae/24k_mono_latent64.json",
        help="Path to the autoencoder config.",
    )
    parser.add_argument(
        "--add-model-ckpt",
        type=str,
        default="./pretrained_models/add_model.pt",
        help="Path to the add-model checkpoint.",
    )
    parser.add_argument(
        "--add-model-vae-ckpt",
        type=str,
        default="./pretrained_models/audio-vae.ckpt",
        help="Path to the add-model VAE checkpoint.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7862, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")

    args = parser.parse_args()

    if not args.gemini_api_key:
        print("❌ Missing API key. Set GEMINI_API_KEY or pass --gemini-api-key.")
        return

    global GEMINI_CLIENT
    GEMINI_CLIENT = GeminiPlannerClient(
        api_key=args.gemini_api_key,
        model=args.gemini_model,
        base_url=args.gemini_base_url,
    )

    global PIPELINE_MODELS
    print("🔄 Loading diffusion editor + add models...")

    try:
        PIPELINE_MODELS = load_pipeline_models(
            diffusion_ckpt=args.diffusion_ckpt,
            autoencoder_ckpt=args.autoencoder_ckpt,
            autoencoder_config_path=args.autoencoder_config,
            diffusion_config_path=args.diffusion_config,
            add_model_ckpt=args.add_model_ckpt,
            add_model_vae_ckpt=args.add_model_vae_ckpt,
        )
        print("✅ Models loaded successfully.")
    except Exception as exc:
        print(f"❌ Failed to load models: {exc}")
        return

    demo = create_gradio_interface()
    print(f"🚀 Launching SmartDJ pipeline on http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=PIPELINE_CSS,
        show_error=True,
    )


if __name__ == "__main__":
    main()
