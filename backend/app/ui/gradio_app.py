"""Simple Gradio chat interface."""

## backend/app/ui/gradio_app.py
import gradio as gr
import requests
import json
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class GradioChatInterface:
    """Gradio chat interface for LLM Switch Chat."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")

    def chat_fn(
        self, message: str, history: List[Tuple[str, str]], use_rag: bool = False
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Chat function for Gradio interface."""
        if not message.strip():
            return history, ""

        try:
            # Convert Gradio history to API format
            api_history = []
            for user_msg, assistant_msg in history:
                api_history.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    api_history.append({"role": "assistant", "content": assistant_msg})

            # Make API request
            response = requests.post(
                f"{self.api_base_url}/chat/",
                json={"message": message, "history": api_history, "use_rag": use_rag},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                assistant_response = result["response"]

                # Add to history
                new_history = history + [(message, assistant_response)]
                return new_history, ""
            else:
                error_msg = f"API Error: {response.status_code}"
                new_history = history + [(message, error_msg)]
                return new_history, ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            new_history = history + [(message, error_msg)]
            return new_history, ""

    def upload_document_fn(self, file_path: str) -> str:
        """Upload document to RAG system."""
        if not file_path:
            return "No file selected"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            response = requests.post(
                f"{self.api_base_url}/rag/upload",
                json={
                    "content": content,
                    "metadata": {"filename": file_path.split("/")[-1]},
                },
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                return result["message"]
            else:
                return f"Upload failed: {response.status_code}"

        except Exception as e:
            return f"Upload error: {str(e)}"

    def get_status_fn(self) -> str:
        """Get API status."""
        try:
            response = requests.get(f"{self.api_base_url}/chat/status", timeout=10)

            if response.status_code == 200:
                result = response.json()
                status_text = f"Backend Available: {result['backend_available']}\n"
                status_text += f"RAG Available: {result['rag']['available']}\n"
                if result["rag"]["available"]:
                    status_text += f"Documents: {result['rag']['document_count']}"
                return status_text
            else:
                return f"Status check failed: {response.status_code}"

        except Exception as e:
            return f"Status error: {str(e)}"

    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""

        with gr.Blocks(title="LLM Switch Chat") as interface:
            gr.Markdown("# LLM Switch Chat")
            gr.Markdown("Multi-backend LLM chat with RAG support")

            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Conversation")

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message", placeholder="Enter your message...", scale=4
                    )
                    use_rag_checkbox = gr.Checkbox(
                        label="Use RAG", value=False, scale=1
                    )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

                # Chat functionality
                def chat_wrapper(message, history, use_rag):
                    return self.chat_fn(message, history, use_rag)

                send_btn.click(
                    chat_wrapper,
                    inputs=[msg_input, chatbot, use_rag_checkbox],
                    outputs=[chatbot, msg_input],
                )

                msg_input.submit(
                    chat_wrapper,
                    inputs=[msg_input, chatbot, use_rag_checkbox],
                    outputs=[chatbot, msg_input],
                )

                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

            with gr.Tab("RAG Management"):
                gr.Markdown("### Upload Documents")

                file_input = gr.File(
                    label="Upload Document (.txt, .md)", file_types=[".txt", ".md"]
                )
                upload_btn = gr.Button("Upload to RAG", variant="primary")
                upload_output = gr.Textbox(label="Upload Status", interactive=False)

                upload_btn.click(
                    self.upload_document_fn,
                    inputs=[file_input],
                    outputs=[upload_output],
                )

                gr.Markdown("### System Status")
                status_btn = gr.Button("Check Status")
                status_output = gr.Textbox(label="System Status", interactive=False)

                status_btn.click(self.get_status_fn, outputs=[status_output])

        return interface


def main():
    """Main function to run Gradio interface."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Switch Chat Gradio Interface")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Gradio server port (default: 7860)"
    )
    parser.add_argument("--share", action="store_true", help="Create public share link")

    args = parser.parse_args()

    # Create chat interface
    chat_interface = GradioChatInterface(api_base_url=args.api_url)
    interface = chat_interface.create_interface()

    # Launch interface
    interface.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
