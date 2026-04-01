from __future__ import annotations

import logging
import os

from configs.schema import AppConfig
from demo.analysis import analyze_live_frame, analyze_uploaded_video, reset_live_outputs
from flow import build_flow_estimator
from training.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def build_interface(config: AppConfig):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gradio is required for the web demo. Install with `pip install gradio`."
        ) from exc

    pipeline = TrainingPipeline(config)
    runtime_summary = pipeline.dry_run_summary()
    backend_names = list(dict.fromkeys([config.flow.backend, *config.demo.enabled_backends]))
    estimators = {name: build_flow_estimator(name, params=config.flow.params) for name in backend_names}
    backend_summaries = {name: estimator.summary() for name, estimator in estimators.items()}
    selectable_backends = [
        name for name in backend_names if backend_summaries.get(name, {}).get("available", False)
    ]
    if not selectable_backends:
        selectable_backends = [config.flow.backend]
    default_backend = config.flow.backend if config.flow.backend in selectable_backends else selectable_backends[0]
    logger.info(
        "Building Gradio interface | title=%s | runtime_device=%s | backends=%s",
        config.demo.title,
        runtime_summary["runtime"]["device"],
        selectable_backends,
    )

    def reset_session(selected_backend: str):
        logger.info("Resetting demo session | backend=%s", selected_backend)
        return reset_live_outputs(selected_backend, runtime_summary, backend_summaries)

    def stream_camera(frame, selected_backend: str, live_state):
        return analyze_live_frame(
            frame=frame,
            backend=selected_backend,
            state=live_state,
            estimators=estimators,
            backend_summaries=backend_summaries,
            runtime_summary=runtime_summary,
        )

    def review_clip(video_path: str | None, selected_backend: str):
        return analyze_uploaded_video(
            video_path=video_path,
            backend=selected_backend,
            estimators=estimators,
            backend_summaries=backend_summaries,
            runtime_summary=runtime_summary,
        )

    initial_live_image, initial_live_status, initial_guidance, initial_metrics, initial_details, initial_state = (
        reset_session(default_backend)
    )
    initial_clip_image, initial_clip_status, initial_clip_guidance, initial_clip_metrics, initial_clip_details = (
        review_clip(None, default_backend)
    )

    with gr.Blocks(title=config.demo.title) as interface:
        gr.Markdown(
            f"# {config.demo.title}\n"
            "ActionFlow currently behaves as a live motion-analysis and capture-debugging workbench for action-recognition experiments. "
            "It does not run a trained classifier yet, but it now gives immediate feedback about framing, motion strength, lighting, backend availability, runtime state, camera shake, and a heuristic action probe.\n\n"
            "Supported heuristic action probes: `Wave`, `Arm Raise`, `Side Step`, `Squat / Sit-to-Stand`, `Jumping Motion`, plus `General Motion` and `Camera Motion`."
        )

        with gr.Row():
            gr.JSON(label="Runtime Snapshot", value=runtime_summary["runtime"])
            gr.JSON(label="Backend Inventory", value=backend_summaries)

        with gr.Tab("Live Capture Lab"):
            live_state = gr.State(initial_state)
            with gr.Row():
                with gr.Column(scale=5):
                    webcam_backend = gr.Dropdown(
                        choices=selectable_backends,
                        value=default_backend,
                        label="Motion Backend",
                    )
                    webcam_input = gr.Image(
                        label="Live Camera",
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        webcam_options=gr.WebcamOptions(mirror=True),
                    )
                    live_output = gr.Image(
                        label="Live Analysis",
                        value=initial_live_image,
                        type="numpy",
                        streaming=True,
                    )
                with gr.Column(scale=4):
                    live_status = gr.Markdown(value=initial_live_status)
                    live_guidance = gr.Markdown(value=initial_guidance)
                    live_metrics = gr.JSON(label="Live Metrics", value=initial_metrics)
                    live_details = gr.Code(label="Live Debug Trace", value=initial_details, language="json")
                    reset_button = gr.Button("Reset Live Session")

            webcam_input.stream(
                fn=stream_camera,
                inputs=[webcam_input, webcam_backend, live_state],
                outputs=[live_output, live_status, live_guidance, live_metrics, live_details, live_state],
                concurrency_limit=1,
                stream_every=max(config.demo.live_frame_interval_ms / 1000.0, 0.08),
                time_limit=300,
                show_progress="hidden",
            )
            webcam_input.clear(
                fn=reset_session,
                inputs=[webcam_backend],
                outputs=[live_output, live_status, live_guidance, live_metrics, live_details, live_state],
                show_progress="hidden",
            )
            webcam_backend.change(
                fn=reset_session,
                inputs=[webcam_backend],
                outputs=[live_output, live_status, live_guidance, live_metrics, live_details, live_state],
                show_progress="hidden",
            )
            reset_button.click(
                fn=reset_session,
                inputs=[webcam_backend],
                outputs=[live_output, live_status, live_guidance, live_metrics, live_details, live_state],
                show_progress="hidden",
            )

        with gr.Tab("Clip Review"):
            with gr.Row():
                with gr.Column(scale=5):
                    upload_backend = gr.Dropdown(
                        choices=selectable_backends,
                        value=default_backend,
                        label="Motion Backend",
                    )
                    upload_input = gr.Video(label="Upload Video", sources=["upload"])
                    upload_output = gr.Image(
                        label="Peak Motion Preview",
                        value=initial_clip_image,
                        type="numpy",
                    )
                with gr.Column(scale=4):
                    upload_status = gr.Markdown(value=initial_clip_status)
                    upload_guidance = gr.Markdown(value=initial_clip_guidance)
                    upload_metrics = gr.JSON(label="Clip Metrics", value=initial_clip_metrics)
                    upload_details = gr.Code(
                        label="Clip Debug Trace",
                        value=initial_clip_details,
                        language="json",
                    )

            upload_input.change(
                fn=review_clip,
                inputs=[upload_input, upload_backend],
                outputs=[upload_output, upload_status, upload_guidance, upload_metrics, upload_details],
                show_progress="minimal",
            )
            upload_backend.change(
                fn=review_clip,
                inputs=[upload_input, upload_backend],
                outputs=[upload_output, upload_status, upload_guidance, upload_metrics, upload_details],
                show_progress="minimal",
            )

        with gr.Tab("How To Collect Better Data"):
            gr.Markdown(
                "### Capture Checklist\n"
                "1. Keep the camera fixed and keep background motion low.\n"
                "2. Make sure your upper body or full body stays visible through the whole action.\n"
                "3. Start with a still pose, perform one action cleanly, then return to stillness.\n"
                "4. Record short clips with one action label per clip.\n"
                "5. Use strong front lighting so hands, elbows, and torso separate from the background.\n\n"
                "### Good First Actions\n"
                "Wave, arm raise, squat, side step, lunge, sit-to-stand, marching in place, or a single jumping jack.\n\n"
                "### Why This View Is Useful\n"
                "This app helps you debug framing, scene quality, motion visibility, and backend readiness before spending time on model training."
            )

    interface.queue(default_concurrency_limit=1)
    return interface


def launch_demo(config: AppConfig) -> None:
    interface = build_interface(config)
    server_name = os.getenv("GRADIO_SERVER_HOST", config.demo.host)
    server_port = int(os.getenv("GRADIO_SERVER_PORT", str(config.demo.port)))
    logger.info(
        "Launching demo | host=%s | port=%s | share=%s",
        server_name,
        server_port,
        config.demo.share,
    )
    interface.launch(server_name=server_name, server_port=server_port, share=config.demo.share)
