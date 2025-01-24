## WeInfer: 

### Overview

This repostory stores the source code of WeInfer, a Web-based LLM inference framework developed on the top of [WebLLM](https://github.com/mlc-ai/web-llm).

**Features**

- Seamlessly integrated with WebLLM and its advanced optimizations like kernel tuning. Support all models with MLC formats.
- Additional speedup compared with WebLLM, benefiting from WebGPU buffer reuse and asychronous pipeline techniques.

### Build

WeInfer is built on the top of WebLLM. The modified source code is in the folder `web-llm`. The approach to build WeInfer is almost the same with building the WebLLM library, which can be referred to https://github.com/mlc-ai/web-llm?tab=readme-ov-file#build-webllm-package-from-source. 

```bash
# build tvm/relax webruntime （tvmjs@0.17.0-dev0）
cd web-llm/3rdparty/tvm-unity/web
make clean && make
npm run bundle

# build web-llm @0.2.46
cd web-llm
npm run build
```

The built lib will be at `web-llm/3rdparty/tvm-unity/web/lib` and `web-llm/lib`.

This repostory also provided prebuilt_lib in the folder `built_lib`, which is defaultly be used by `demo_page`.

### Usage

The folder `demo_page/` contains a demo page with all fundamental functionalities for running LLM within browsers. Start it by:

```bash
cd demo_page
npm install
npm run dev
```
This will create a server running at https://localhost:8885. Visit this site to use WeInfer.

WeInfer can load model weights from URL linked to huggingface or a local model server. You can change this by modifying `appConfig.model_list` and `modelServerUrl` in the file `demo_page/src/get_started.ts`.


Defaultly WeInfer utilizes model server in the folder `model_server/`. Put your certificate `server.crt` and private key `key.pem` in `model_server/`, and then start the model server at https://localhost:8886 by:
```bash
cd model_server
npm install
npm run dev
```

To use more LLM models, modify `appConfig.model_list` in the file `demo_page/src/get_started.ts`, adding URL of your custom model and model lib in the MLC format.