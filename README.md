# WeInfer: Unleashing the Power of WebGPU on LLM Inference in Web Browsers

## Overview

This repostory stores the source code of WeInfer, a Web-based LLM inference framework developed on the top of [WebLLM](https://github.com/mlc-ai/web-llm). WebLLM is designed in a WebGPU-centric approach to accelerate LLM inference within the browsers.

![Overview](docs/framework_overview.png)

**Features**

- Seamlessly integrated with WebLLM and its advanced optimizations like kernel tuning. Support all models with MLC formats.
- Additional speedup compared with WebLLM, benefiting from WebGPU buffer reuse and asychronous pipeline techniques.

## Core Implementation Guide

We implement a prototype of WeInfer for based on WebLLM (version 0.2.46). We modify the procedure of WebGPU buffer creating and fetching of the WebLLM runtime. Modified code is seamlessly integrated to WebLLM and is stored at `web-llm/`.

### Build

As this prototype of WeInfer is built on the top of WebLLM, the approach to build WeInfer is almost the same with building the WebLLM library.

**Setup**

The required environment for building WeInfer are the same with building WebLLM runtime, include NPM, emscripten, etc. Please follow the instructions in https://github.com/mlc-ai/web-llm?tab=readme-ov-file#build-webllm-package-from-source to setup the environment.

**Build**

First we need to build TVMjs, which WebLLM runtime depends on:

```bash
# build tvm/relax webruntime （tvmjs@0.17.0-dev0）
cd web-llm/3rdparty/tvm-unity/web
make clean && make
npm run bundle
```

The built lib will be at `web-llm/3rdparty/tvm-unity/web/lib`.

Then we can build the WebLLM runtime by:

```bash
# build web-llm @0.2.46
cd web-llm
npm run build
```

The built lib will be at `web-llm/lib`. Then you can use this built lib `web-llm/lib` like regular WebLLM library.

This repostory also provides prebuilt lib in the folder `built_lib/`, which is defaultly be used by our demo application. `built_lib/` also includes library of WeInfer baseline and buffer reuse baseline for evaluation. Prebuilt library of WeInfer with full optimizations is at `built_lib/web-llm-ours/lib-all-nx-gpu-sample-reb`.

### Usage

You can use the built WeInfer library like any other NPM package. We also provides a demo application in this repository. The folder `demo_page/` contains a demo page with all fundamental functionalities like loading models and executing inference for running LLM within browsers. 

**Usage of Demo Application**

Start this demo page by:

```bash
cd demo_page
npm install
npm run dev
```
This will create a server running at https://localhost:8885. Visit this site to use WeInfer. Defaultly, this demo page loads model weights from a local model server. You should modify `modelServerUrl` in the file `demo_page/src/get_started.ts` to URL of your server. 

The demo page can also load model weights from huggingface. You can change this by modifying `appConfig.model_list` in the file `demo_page/src/get_started.ts`.


**Usage of Model Server**

We provide a simple model server in the folder `model_server/`. Put your certificate `server.crt` and private key `key.pem` in `model_server/`, and then start the model server at https://localhost:8886 by:
```bash
cd model_server
npm install
npm run dev
```

Download the model weights by:
```bash
cd model_server/src/resolve
git clone https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f32-MLC
```
**More Models**

To use more LLM models, modify `appConfig.model_list` in the file `demo_page/src/get_started.ts`, adding URL of your custom model and model lib in the MLC format.