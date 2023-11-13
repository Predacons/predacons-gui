import gradio as gr
import predacons

class WebApp:
    # data preprocessing

    def __read_text(files):
        raw_text = predacons.read_multiple_files(files)
        return raw_text

    def __clean_text(text_data):
        c_text = predacons.clean_text(text_data)
        return c_text

    def __save_input_text(text):
        with open("input_text_tmp.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return "input_text_tmp.txt"
    
    # training
    def __trainer(train_file_path,
        model_name,
        output_dir,
        overwrite_output_dir,
        per_device_train_batch_size,
        num_train_epochs,
        save_steps):
        return predacons.trainer(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

    def __train_model(train_file_path,
        model_name,
        output_dir,
        overwrite_output_dir,
        per_device_train_batch_size,
        num_train_epochs,
        save_steps,progress=gr.Progress()):

        tt = WebApp.__trainer(train_file_path,
        model_name,
        output_dir,
        overwrite_output_dir,
        per_device_train_batch_size,
        num_train_epochs,
        save_steps)

        progress(0, desc="Starting...")
        for epoch in progress.tqdm(range(int(num_train_epochs)), desc="Training..."):
            tt.train()
        progress(None, desc="Training completed!")
        return "Training completed!"
    
    # generate text
    def __history_to_text(history):
        history_text = ""
        for text in history:
            print(text)
            if text[1] is None:
                history_text += "Q: "+text[0] +"\n"
            else:
                history_text += "Q: "+text[0] +"\n"+"A: "+text[1]+"\n"
        return history_text
        

    def __add_text(history, text,enable_history=True):
        if enable_history:
            history = history + [(text, None)]
        else:
            history = [(text, None)]
        print(history)
        return history, gr.Textbox(value="", interactive=False)


    def __bot(history,model,max_len):
        print (history)
        history_text = WebApp.__history_to_text(history)
        print(history_text)
        history[-1][1] = ""
        response = predacons.generate_text(model, history_text, max_len)
        for character in response:
            history[-1][1] += character
            yield history
    

    def __web_page():
        with gr.Blocks() as gui:
            gr.Markdown ("""
            # Predacons""")
            with gr.Tab(label="Train"):
                save_input = None
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Data Preparation")
                        files_path = gr.File(label="Select files",file_count = "multiple")
                        btn = gr.Button("Get Text from the directory")
                        Raw_text = gr.Textbox(label="Raw Text")
                        btn.click(WebApp.__read_text, inputs=[files_path], outputs=[Raw_text])
                        btn2 = gr.Button("Clean up text")
                        Clean_text = gr.Textbox(label="Clean Text")
                        btn2.click(WebApp.__clean_text, inputs=[Raw_text], outputs=[Clean_text])
                        btn3 = gr.Button("Save the input text")
                        save_input = gr.Textbox(label="Save the input text")
                        btn3.click(WebApp.__save_input_text, inputs=[Clean_text], outputs=[save_input])

                    with gr.Column(scale=3):
                        gr.Markdown("## Train")
                        model_name = None
                        with gr.Tab(label = "local model"):
                            model_name = gr.Textbox(label="Model path")
                        with gr.Tab(label = "Huggingface model"):
                            model_name = gr.Dropdown(["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],allow_custom_value=True)
                        output_dir = gr.Textbox(label="Output Directory")
                        overwrite_output_dir = gr.Checkbox(label="Overwrite Output Directory")
                        per_device_train_batch_size = gr.Slider(minimum=1, maximum=64, value=8, label="Per Device Train Batch Size")
                        num_train_epochs = gr.Slider(minimum=1, maximum=100, value=50, label="Number of Train Epochs")
                        save_steps = gr.Slider(minimum=1, maximum=100000, value=50000, label="Save Steps")
                        btn = gr.Button("Train")
                        completed = gr.Textbox(label="Training completed")
                        btn.click(WebApp.__train_model, inputs=[save_input,model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps],outputs=[completed])
                    
            with gr.Tab(label="Chat"):
                model_name = None
                max_len = None
                
                with gr.Row():
                    with gr.Tab(label = "local model"):
                        model_name = gr.Textbox(label="Model path")
                    with gr.Tab(label = "Huggingface model"):
                        model_name = gr.Dropdown(["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],allow_custom_value=True)
                    max_len = gr.Slider(minimum=1, maximum=500, value=100, label="max length of the response")
                with gr.Row():
                    enable_history = gr.Checkbox(label="Enable History")
                chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                # avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
                )

                with gr.Row():
                    txt = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False,
                    )
                txt_msg = txt.submit(WebApp.__add_text, [chatbot, txt, enable_history], [chatbot, txt], queue=False).then(
                    WebApp.__bot, [chatbot,model_name,max_len], chatbot, api_name="bot_response"
                )
                txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
                clear = gr.ClearButton([txt, chatbot])

        return gui

    def launch():
        webapp = WebApp.__web_page()
        webapp.launch()