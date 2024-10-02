import gradio as gr
import predacons
import openai
import os

class WebApp:
    # data preprocessing
    MODEL = None
    TOKENIZER = None
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
        
    # generate chat
    def __history_to_chat(history):
        chat_body = []
        for text in history:
            print(text)
            if text[1] is None:
                chat_body.append({"role": "user", "content": text[0]})
            else:
                chat_body.append({"role": "user", "content": text[0]})
                chat_body.append({"role": "user", "content": text[0]})
        return chat_body

    def __load_model(model_path,gguf_file=None,auto_quantize=None):
        global MODEL, TOKENIZER
        MODEL = predacons.load_model(model_path,gguf_file=gguf_file,auto_quantize =auto_quantize)
        TOKENIZER = predacons.load_tokenizer(model_path)
        # return [model, tokenizer]
    def __add_text(history, text,enable_history=True):
        if enable_history:
            history = history + [(text, None)]
        else:
            history = [(text, None)]
        print(history)
        return history, gr.Textbox(value="", interactive=False)


    def __bot(history,max_len,temperature):
        print (history)
        chat_body = WebApp.__history_to_chat(history)
        print(chat_body)
        history[-1][1] = ""
        global MODEL, TOKENIZER
        # response = predacons.generate_text(model, history, max_len)
        response = predacons.chat_generate(
            model = MODEL,
            sequence = chat_body,
            max_length = max_len,
            tokenizer = TOKENIZER,
            trust_remote_code = True,
            do_sample=True,
            temperature = temperature
        )
        for character in response:
            history[-1][1] += character
            yield history
    
    def __create_openai_client(gpt_api_version,gpt_api_key = None,gpt_azure_endpoint = None,is_azure_openai = False):
        print("crating open ai client") 
        client = None
        if(not(gpt_api_key == None or gpt_api_key == "")):
            os.environ['AZURE_OPENAI_API_KEY'] = gpt_api_key
            os.environ['OPENAI_API_KEY'] = gpt_api_key
        if(is_azure_openai):
            client = openai.AzureOpenAI(
                    api_version=gpt_api_version,
                    azure_endpoint=gpt_azure_endpoint,
                )
        else:
            client= openai.OpenAI()
        return client
    
    def __generate_text_data_openai(gpt_model,prompt,no_of_output,temp,gpt_api_version,gpt_api_key = None,gpt_azure_endpoint = None):
        is_azure_openai = True
        if(gpt_azure_endpoint == None or gpt_azure_endpoint == ""):
            is_azure_openai = False
        client = WebApp.__create_openai_client(gpt_api_version,gpt_api_key,gpt_azure_endpoint,is_azure_openai)
        return predacons.generate_text_data_source_openai(client,gpt_model,prompt,no_of_output,temp)
    
    def __generate_text_data_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
        return predacons.generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=trust_remote_code)


    def __web_page():
        with gr.Blocks() as gui:
            gr.Markdown ("""
            <table>
                <tr >
                    <td style="border: none; padding: 0;">
                    <a href="https://github.com/Predacons"><img src="https://i.postimg.cc/YSzMP1M8/pngegg-1-1.png" width="60px" height="60px"></a>
                    </td>
                    <td style="border: none; padding: 10; vertical-align: middle;">
                    <a href="https://github.com/Predacons" style="text-decoration: none;"><H1>Predacons<H1></a>
                    </td>
                </tr>
            </table>""")
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
                model_name1 = None
                max_len = None
                model=None
                tokenizer = None
                model_output = None
                enable_history = None
                temp = None
                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label = "local model"):
                            model_name1 = gr.Textbox(label="Model path")
                            load_model_btn1 = gr.Button("Load Model")
                        with gr.Tab(label = "Huggingface model"):
                            model_name2 = gr.Dropdown(["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],allow_custom_value=True)
                            load_model_btn2 = gr.Button("Load Model")
                    with gr.Column():
                        max_len = gr.Slider(minimum=1, maximum=5000, value=500, label="max length of the response")
                        temp = gr.Slider(minimum=0, maximum=2, value=0.3, label="temprerature of the model")
                        enable_history = gr.Checkbox(label="Enable History")
                        model_output = gr.Textbox(label="Model Output", interactive=False)
                # model1 = load_model_btn2.click(WebApp.__load_model, inputs=[model_name])
                load_model_btn1.click(WebApp.__load_model, inputs=[model_name1])
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
                    WebApp.__bot, [chatbot,max_len,temp], chatbot, api_name="bot_response"
                )
                txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
                clear = gr.ClearButton([txt, chatbot])
            with gr.Tab(label="Training Data Creation"):
                with gr.Row():
                    model_name
                    prompt = None
                    gpt_api_version = None
                    gpt_azure_endpoint = None
                    gpt_api_key = None
                    gpt_model = None
                    temperature = None
                    no_of_results = None
                    training_data = None
                    client = None
                    with gr.Tab(label = "OpenAI"):
                        with gr.Tab(label = "OpenAI"):
                            gpt_api_version = gr.Textbox(label="gpt_api_version")
                            gpt_api_key = gr.Textbox(label="gpt_api_key")
                            btn = gr.Button("Create OpenAI Client")
                            gpt_model = gr.Textbox(label="gpt_model")
                            temperature = gr.Slider(minimum=0, maximum=1, value=0.5, label="Temperature")
                            no_of_results = gr.Slider(minimum=1, maximum=100, value=10,step=1, label="No of outputs")
                            prompt = gr.Textbox(label="Prompt")
                            gen_btn = gr.Button("generate training data set")
                            training_data = gr.Textbox(label="training data")
                            gen_btn.click(WebApp.__generate_text_data_openai, inputs=[gpt_model,prompt,no_of_results,temperature,gpt_api_version,gpt_api_key],outputs=[training_data])
                        
                        with gr.Tab(label = "Azure OpenAI"):
                            gpt_api_version = gr.Textbox(label="gpt_api_version")
                            gpt_azure_endpoint = gr.Textbox(label="gpt_azure_endpoint")
                            gpt_api_key = gr.Textbox(label="gpt_api_key")
                            gpt_model = gr.Textbox(label="gpt_model")
                            temperature = gr.Slider(minimum=0, maximum=1, value=0.5, label="Temperature")
                            no_of_results = gr.Slider(minimum=1, maximum=100, value=10,step=1, label="No of outputs")
                            prompt = gr.Textbox(label="Prompt")
                            gen_btn = gr.Button("generate training data set")
                            training_data = gr.Textbox(label="training data")
                            gen_btn.click(WebApp.__generate_text_data_openai, inputs=[gpt_model,prompt,no_of_results,temperature,gpt_api_version,gpt_api_key,gpt_azure_endpoint],outputs=[training_data])
                    
                    with gr.Tab(label = "LLM model"):
                        model_name
                        with gr.Tab(label = "local model"):
                            model_name = gr.Textbox(label="Model path")
                        with gr.Tab(label = "Huggingface model"):
                            model_name = gr.Dropdown(["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],allow_custom_value=True)
                        max_len = gr.Slider(minimum=1, maximum=500, value=100, label="max length of the response")
                        no_of_results = gr.Slider(minimum=1, maximum=100, value=10,step=1, label="No of outputs")
                        prompt = gr.Textbox(label="Prompt")
                        trust_remote_code = gr.Checkbox(label="enable trust remote code")
                        gen_btn = gr.Button("generate training data set")
                        training_data1 = gr.Textbox(label="training data")
                        gen_btn.click(WebApp.__generate_text_data_llm, inputs=[model_name, prompt, max_len,no_of_results,trust_remote_code],outputs=[training_data1])


                            
            gr.Markdown ("""
                        <p style="text-align: center; font-size: small;">Powered by <a href="https://github.com/Predacons">Predacons</a> </p>
                        <p style="text-align: center; font-size: small;"><a href="https://github.com/shouryashashank">Shourya Shashank</a> </p> """)                
                    
        return gui

    def launch():
        webapp = WebApp.__web_page()
        webapp.launch()


WebApp.launch()