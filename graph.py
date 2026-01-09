from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from typing_extensions import Annotated
import operator
import base64
import random
import os
import json
from google.genai import Client
from google.genai.types import Part, Content

llm = init_chat_model("openai:gpt-4o-mini")
gemini_client = Client(api_key=os.getenv("GOOGLE_API_KEY"))


class State(TypedDict):

    video_file: str
    audio_file: str
    screenshots: list[str]
    chosen_screenshot: str
    chosen_screenshot_index: int
    whisper_prompt: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[str], operator.add]
    final_summary: str
    user_feedback: str
    chosen_thumbnail: str
    final_thumbnail: str
    evaluator_feedback: str
    evaluation_passed: bool
    hd_model: str
    evaluation_failures: int


def extract_audio(state: State):
    output_file = state["video_file"].replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file,
    ]
    subprocess.run(command)
    return {
        "audio_file": output_file,
    }


def extract_screenshots(state: State):
    # 1. FFprobe로 영상 길이 확인
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        state["video_file"],
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())

    # 2. 랜덤 타임스탬프 생성 (5장)
    num_screenshots = 5
    timestamps = sorted(random.sample(range(1, int(duration) - 1), num_screenshots))

    # 3. 각 타임스탬프에서 스크린샷 추출
    screenshot_files = []
    for i, ts in enumerate(timestamps):
        output_file = f"screenshot_{i+1}.jpg"
        cmd = [
            "ffmpeg",
            "-ss",
            str(ts),
            "-i",
            state["video_file"],
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            output_file,
        ]
        subprocess.run(cmd)
        screenshot_files.append(output_file)

    return {
        "screenshots": screenshot_files,
    }


def choose_screenshot(state: State):
    screenshots = state["screenshots"]
    prompt_lines = ["Select a screenshot number or enter 0 to resample:"]
    for idx, path in enumerate(screenshots, start=1):
        prompt_lines.append(f"{idx}. {path}")
    prompt_lines.append("0. None of these (resample)")

    answer = interrupt({"chosen_screenshot": "\n".join(prompt_lines)})
    raw_choice = (
        answer if isinstance(answer, str) else answer.get("chosen_screenshot", "")
    )

    try:
        choice = int(raw_choice)
    except (TypeError, ValueError):
        choice = 0

    if choice < 0 or choice > len(screenshots):
        choice = 0

    selected = screenshots[choice - 1] if choice > 0 else ""

    return {
        "chosen_screenshot": selected,
        "chosen_screenshot_index": choice,
    }


def route_screenshot_choice(state: State):
    return (
        "extract_screenshots"
        if state.get("chosen_screenshot_index", 0) == 0
        else "get_transcription_prompt"
    )


def get_transcription_prompt(state: State):
    answer = interrupt(
        {
            "whisper_prompt": "전사를 위한 Whisper 프롬프트를 입력하세요 (비디오 주제 관련 키워드, 쉼표로 구분)"
        }
    )
    return {
        "whisper_prompt": (
            answer if isinstance(answer, str) else answer.get("whisper_prompt", "")
        ),
    }


def transcribe_audio(state: State):
    client = OpenAI()
    with open(state["audio_file"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="ko",
            prompt=state.get("whisper_prompt", ""),
        )
        return {
            "transcription": transcription,
        }


def dispatch_summarizers(state: State):
    transcription = state["transcription"]
    chunks = []
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append({"id": i + 1, "chunk": chunk})
    return [Send("summarize_chunk", chunk) for chunk in chunks]


def summarize_chunk(chunk):
    chunk_id = chunk["id"]
    chunk = chunk["chunk"]

    response = llm.invoke(
        f"""
        Please summarize the following text.

        Text: {chunk}
        """
    )
    summary = f"[Chunk {chunk_id}] {response.content}"
    return {
        "summaries": [summary],
    }


def mega_summary(state: State):

    all_summaries = "\n".join(state["summaries"])

    prompt = f"""
        You are given multiple summaries of different chunks from a video transcription.

        Please create a comprehensive final summary that combines all the key points.

        Individual summaries:

        {all_summaries}
    """

    response = llm.invoke(prompt)

    return {
        "final_summary": response.content,
    }


def dispatch_artists(state: State):
    selected_screenshot = state.get("chosen_screenshot") or state["screenshots"][0]
    return [
        Send(
            "generate_thumbnails",
            {
                "id": i,
                "summary": state["final_summary"],
                "screenshot": selected_screenshot,
            },
        )
        for i in [1, 2, 3]
    ]


def generate_thumbnails(args):
    concept_id = args["id"]
    summary = args["summary"]
    screenshot = args["screenshot"]

    # 1. 스크린샷 선택 및 로드
    with open(screenshot, "rb") as img_file:
        image_data = img_file.read()

    # 2. 썸네일 프롬프트 생성
    thumbnail_prompt = f"""
    Create a professional YouTube thumbnail based on this image from the video.

    Video Summary: {summary}

    Transform this screenshot into an eye-catching thumbnail:
    - Keep the main visual elements and composition from the original image
    - Enhance colors to make them pop and grab attention
    - Add bold, readable text overlays with generous padding from edges
    - Apply professional effects like subtle vignetting or sharpening
    - Ensure it works well at small sizes
    - Make it click-worthy while staying true to the video content
    """

    # 3. Gemini 2.5 Flash Image로 썸네일 생성
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            Content(
                parts=[
                    Part(text=thumbnail_prompt),
                    Part(
                        inline_data={
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_data).decode(),
                        }
                    ),
                ]
            )
        ],
    )

    # 4. 생성된 이미지 저장
    filename = f"thumbnail_{concept_id}.png"

    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            # inline_data로 접근
            if hasattr(part, "inline_data") and part.inline_data:
                # Gemini는 bytes로 반환
                if isinstance(part.inline_data.data, bytes):
                    image_bytes = part.inline_data.data
                else:
                    image_bytes = base64.b64decode(part.inline_data.data)

                with open(filename, "wb") as file:
                    file.write(image_bytes)
                break

    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [filename],
    }


def human_feedback(state: State):
    answer = interrupt(
        {
            "chosen_thumbnail": "Which thumbnail do you like the most?",
            "user_feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
        }
    )
    user_feedback = answer["user_feedback"]
    chosen_thumbnail_index = answer["chosen_thumbnail"]
    return {
        "user_feedback": user_feedback,
        "chosen_thumbnail": state["thumbnail_sketches"][chosen_thumbnail_index - 1],
    }


def generate_hd_thumbnail(state: State):

    chosen_thumbnail = state["chosen_thumbnail"]
    user_feedback = state["user_feedback"]
    evaluator_feedback = state.get("evaluator_feedback", "")
    model_name = state.get("hd_model", "gemini-2.5-flash-image")

    # 1. 선택한 썸네일 로드
    with open(chosen_thumbnail, "rb") as img_file:
        image_data = img_file.read()

    evaluator_block = ""
    if evaluator_feedback:
        evaluator_block = f"\n\nEVALUATOR FEEDBACK:\n{evaluator_feedback}\n"

    # 2. 최종 썸네일 프롬프트 생성
    final_thumbnail_prompt = f"""
    You are a professional YouTube thumbnail designer. Enhance this thumbnail based on user feedback.

    USER FEEDBACK:
    {user_feedback}
    {evaluator_block}

    Create a high-quality final thumbnail that:
    - Maintains the core visual elements and composition from the original image
    - Implements the user's specific feedback requests
    - Adds professional enhancements:
        * High contrast and bold visual elements
        * Clear focal points that draw the eye
        * Professional lighting and composition
        * Optimal text placement with generous padding from edges
        * Colors that pop and grab attention
        * Works well at small thumbnail sizes
        * Adequate white space/padding between text and image borders
    """

    # 3. Gemini 2.5 Flash Image로 최종 썸네일 생성
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=[
            Content(
                parts=[
                    Part(text=final_thumbnail_prompt),
                    Part(
                        inline_data={
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_data).decode(),
                        }
                    ),
                ]
            )
        ],
    )

    # 4. 최종 이미지 저장
    filename = "thumbnail_final.png"

    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                if isinstance(part.inline_data.data, bytes):
                    image_bytes = part.inline_data.data
                else:
                    image_bytes = base64.b64decode(part.inline_data.data)

                with open(filename, "wb") as file:
                    file.write(image_bytes)
                break

    return {
        "final_thumbnail": filename,
    }


class EvaluationOutput(TypedDict):
    passed: bool
    typo_issues: list[str]
    focus_issue: str


def _get_structured_value(response, key, default):
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def evaluate_thumbnail(state: State):
    filename = state.get("final_thumbnail", "thumbnail_final.png")
    with open(filename, "rb") as img_file:
        image_data = img_file.read()

    evaluation_prompt = """
    You are a strict QA evaluator for YouTube thumbnails.

    Check the image for:
    1) Typos or misspellings in any visible text.
    2) Whether the main subject is clear, sharp, and visually prominent.

    """

    structured_llm = llm.with_structured_output(EvaluationOutput)
    image_b64 = base64.b64encode(image_data).decode()
    response = structured_llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": evaluation_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ]
            )
        ]
    )

    passed = _get_structured_value(response, "passed", False)
    typo_issues = _get_structured_value(response, "typo_issues", []) or []
    focus_issue = _get_structured_value(response, "focus_issue", "") or ""

    feedback_parts = []
    if typo_issues:
        feedback_parts.append("Typos: " + "; ".join(typo_issues))
    if focus_issue:
        feedback_parts.append("Focus issue: " + focus_issue)

    evaluator_feedback = "\n".join(feedback_parts).strip()
    if not passed and not evaluator_feedback:
        evaluator_feedback = "Fix any text errors and improve subject clarity/focus."

    failure_count = state.get("evaluation_failures", 0)
    if not passed:
        failure_count += 1

    return {
        "evaluation_passed": passed,
        "evaluator_feedback": evaluator_feedback,
        "hd_model": (
            "gemini-3-pro-image-preview"
            if not passed
            else state.get("hd_model", "gemini-2.5-flash-image")
        ),
        "evaluation_failures": failure_count,
    }


def route_evaluation(state: State):
    if state.get("evaluation_passed"):
        return END

    if state.get("evaluation_failures", 0) >= 2:
        return END

    return "generate_hd_thumbnail"


graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("extract_screenshots", extract_screenshots)
graph_builder.add_node("choose_screenshot", choose_screenshot)
graph_builder.add_node("get_transcription_prompt", get_transcription_prompt)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)
graph_builder.add_node("evaluate_thumbnail", evaluate_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "extract_screenshots")
graph_builder.add_edge("extract_screenshots", "choose_screenshot")
graph_builder.add_conditional_edges(
    "choose_screenshot",
    route_screenshot_choice,
    ["extract_screenshots", "get_transcription_prompt"],
)
graph_builder.add_edge("get_transcription_prompt", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", "evaluate_thumbnail")
graph_builder.add_conditional_edges(
    "evaluate_thumbnail", route_evaluation, ["generate_hd_thumbnail", END]
)

graph = graph_builder.compile(name="mr_thumbs")
