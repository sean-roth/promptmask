"""
PromptMask - AI-powered video masking using natural language
Main application entry point with Gradio interface
"""

import gradio as gr
import logging
from sam3 import SAM3ModelLoader, SAM3Inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptMaskApp:
    """Main application class for PromptMask."""

    def __init__(self):
        """Initialize PromptMask application."""
        self.model_loader = SAM3ModelLoader()
        self.inference_engine = None
        self.model_loaded = False

    def load_model(self):
        """
        Load SAM 3 model with progress display.

        Returns:
            str: Status message
        """
        try:
            status_text = "📥 Downloading SAM 3 model (~3GB)...\n\n"
            status_text += "This only happens on first run.\n"
            status_text += "Subsequent runs will load instantly from cache.\n\n"
            status_text += "Please wait..."

            yield status_text

            # Load model and processor
            self.model_loader.load_model()
            self.inference_engine = SAM3Inference(self.model_loader)
            self.model_loaded = True

            success_msg = "✅ Model loaded successfully!\n\n"
            success_msg += f"Device: {self.model_loader.get_device()}\n"
            success_msg += "Ready to process videos."

            yield success_msg

        except Exception as e:
            error_msg = f"❌ Failed to load model: {str(e)}\n\n"
            error_msg += "Make sure you're authenticated:\n"
            error_msg += "1. Run: huggingface-cli login\n"
            error_msg += "2. Request access: https://huggingface.co/facebook/sam3\n"
            error_msg += "3. Wait for approval (~24 hours)\n"

            yield error_msg

    def process_video_placeholder(self, video_file, text_prompt, quality_preset):
        """
        Placeholder for video processing functionality.
        This will be implemented in future iterations.

        Args:
            video_file: Uploaded video file
            text_prompt: Text description of what to segment
            quality_preset: Quality setting (Fast/Balanced/Accurate)

        Returns:
            str: Status message
        """
        if not self.model_loaded:
            return "❌ Please load the model first!"

        return (
            f"🚧 Video processing coming soon!\n\n"
            f"Video: {video_file.name if video_file else 'None'}\n"
            f"Prompt: '{text_prompt}'\n"
            f"Quality: {quality_preset}\n\n"
            f"This feature is under development."
        )


def create_interface():
    """
    Create and configure Gradio interface.

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    app = PromptMaskApp()

    with gr.Blocks(title="PromptMask - AI Video Masking") as interface:
        gr.Markdown(
            """
            # PromptMask
            ### AI-powered video masking using natural language

            Describe what you want to segment, get production-ready masks for Adobe workflows.
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1️⃣ Load Model")
                load_btn = gr.Button("Load SAM 3 Model", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Click 'Load SAM 3 Model' to begin",
                    lines=10,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### 2️⃣ Process Video")
                video_input = gr.Video(label="Upload Video")
                text_prompt = gr.Textbox(
                    label="What should I segment?",
                    placeholder="e.g., 'person speaking' or 'product on desk'",
                    lines=2
                )
                quality_preset = gr.Radio(
                    choices=["Fast", "Balanced", "Accurate"],
                    value="Balanced",
                    label="Quality Preset"
                )
                process_btn = gr.Button("Process Video", variant="primary", size="lg")
                process_status = gr.Textbox(
                    label="Processing Status",
                    value="Upload a video and enter a prompt to begin",
                    lines=10,
                    interactive=False
                )

        # Event handlers
        load_btn.click(
            fn=app.load_model,
            inputs=[],
            outputs=[model_status]
        )

        process_btn.click(
            fn=app.process_video_placeholder,
            inputs=[video_input, text_prompt, quality_preset],
            outputs=[process_status]
        )

        gr.Markdown(
            """
            ---
            ### Getting Started
            1. **Load Model**: Click the button to download and load SAM 3 (~3GB, first run only)
            2. **Upload Video**: Drag & drop or click to select your video file
            3. **Enter Prompt**: Describe what you want to segment (e.g., "person speaking")
            4. **Process**: Click to generate masks (coming soon)

            ### Documentation
            - [Installation Guide](INSTALLATION.md)
            - [Quick Start](QUICKSTART.md)
            - [Architecture](docs/ARCHITECTURE.md)

            ### Need Help?
            - Check [Troubleshooting](docs/TROUBLESHOOTING.md)
            - Report issues on [GitHub](https://github.com/sean-roth/promptmask/issues)
            """
        )

    return interface


def main():
    """Main entry point for PromptMask application."""
    logger.info("Starting PromptMask application...")

    # Create and launch interface
    interface = create_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
