import gradio as gr
from rag import answer_question

def chat_fn(message, history):
    answer, sources = answer_question(message)
    if sources:
        answer += "\n\nSources:\n" + "\n".join(sources)
    return answer

demo = gr.ChatInterface(
    fn=chat_fn,
    description="Ask questions about electronics datasheets.",   
)
examples = [
    "what is operating voltage of STM32?"
    ]

if __name__ == "__main__":
    demo.launch()