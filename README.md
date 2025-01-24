## WeInfer: 

### Overview

This repostory stores the source code of WeInfer, a Web-based LLM inference framework developed on the top of WebLLM.

**Features**

- Seamlessly integrated with WebLLM and its advanced optimizations like kernel tuning. Support all models with MLC formats.
- Additional speedup compared with WebLLM, benefiting from WebGPU buffer reuse and asychronous pipeline techniques.

### Usage

The folder `demo_page/` contains a demo page with all fundamental functionalities for running LLM within browsers. Start it by:

```bash
cd demo_page
npm run dev
```
This will create a server running at https://localhost:8885. Visit this site and use WeInfer.

WeInfer can load model weights from URL linked to huggingface or a local model server. You can change this by modifying `appConfig.model_list` and `modelServerUrl` in the file `demo_page/src/get_started.ts`.


Defaultly WeInfer utilizes model server in the folder `model_server/`. Start the model server at https://localhost:8886 by:
```bash
cd model_server
npm run dev
```

To use more LLM models, modify `appConfig.model_list` in the file `demo_page/src/get_started.ts`, adding URL of your custom model and model lib in the MLC format.