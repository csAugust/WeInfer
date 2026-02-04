import * as webllm_origin from "@mlc-ai/web-llm";
import * as webllm_cache from "@mlc-ai/web-llm-ours-1x";
import * as webllm from "@mlc-ai/web-llm-ours-nx-gpu-sample-reb";

// function sendResultToServer(result: any, fileName: string) {
//   let xhr = new XMLHttpRequest();
//   let url = "your server";
//   xhr.open("POST", url, true);
//   xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
//   xhr.onreadystatechange = function () {
//     if (xhr.readyState === 4 && xhr.status === 200) {
//       console.log(result);
//     }
//   };
//   let sendData = {
//     fileName: fileName,
//     data: result,
//   };
//   xhr.send(JSON.stringify(sendData));
// }

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

class Engine {
  private initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  public appConfig: webllm.AppConfig;
  public originAppConfig: webllm_origin.AppConfig;
  public engine: webllm.MLCEngineInterface;
  public originEngine: webllm_origin.MLCEngine;

  public selectedLib: number = 3;
  public x: number = 4;
  public selectedModel: string = "SmolLM-135M-Instruct-q0f32-MLC";
  public originModel: string = "SmolLM2-135M-Instruct-q0f32-MLC";
  public isModelLoaded: boolean = false;
  public modelVersion = "v0_2_48";
  public modelLibURLPrefix = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/";
  public modelServerUrl = "localhost:8886"

  constructor() {
    this.appConfig = webllm.prebuiltAppConfig;
    this.appConfig.model_list = [
      {
        // model: "https://huggingface.co/mlc-ai/Qwen2-1.5B-Instruct-q4f32_1-MLC",
        model: `http://${this.modelServerUrl}/resolve/Qwen2-1.5B-Instruct-q4f32_1-MLC/`,
        model_id: "Qwen2-1.5B-Instruct-q4f32_1-MLC",
        model_lib: `http://${this.modelServerUrl}/wasm_libs/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm`,
        // model_lib:
        //   this.modelLibURLPrefix +
        //   this.modelVersion +
        //   "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
        low_resource_required: true,
        vram_required_MB: 1888.97,
        overrides: {
          context_window_size: 4096,
        },
      },
      {
        model: `http://${this.modelServerUrl}/resolve/SmolLM-135M-Instruct-q0f32-MLC`,
        model_id: "SmolLM-135M-Instruct-q0f32-MLC",
        model_lib: `http://${this.modelServerUrl}/wasm_libs/SmolLM-135M-Instruct-q0f32-ctx2k_cs1k-webgpu.wasm`,
        vram_required_MB: 629.38,
        low_resource_required: true,
        overrides: {
          context_window_size: 2048,
        },
      },
    ]
    this.appConfig.useIndexedDBCache = true;

    // Configure origin (latest web-llm) to use SmolLM2 from prebuilt config (HuggingFace)
    this.originAppConfig = webllm_origin.prebuiltAppConfig;
    this.originAppConfig.useIndexedDBCache = true;

    this.loadEngine(this.selectedLib);
  }

  loadEngine(selectedValue: number) {
    if (this.engine) {
      this.engine.unload();
    }
    if (this.originEngine) {
      this.originEngine.unload();
    }
    this.selectedLib = selectedValue;
    console.log(selectedValue);

    if (selectedValue == 1) {
      this.load_origin();
      console.log('load engine webllm_origin');
    } else if (selectedValue == 2) {
      this.load_cache();
      console.log('load engine webllm_cache');
    } else if (selectedValue == 3) {
      this.load_ours();
      console.log('load engine webllm_all');
    }
    this.isModelLoaded = false;
  }

  public updateX(x: number) {
    this.x = x;
    if (this.selectedLib !== 1) {
      this.engine!.getPipeline().reconstructSampler(x);
    }
  }

  private load_ours() {
    this.engine = new webllm.MLCEngine({
      appConfig: this.appConfig, // if do not specify, we use webllm.prebuiltAppConfig
      initProgressCallback: this.initProgressCallback,
    });
  }

  private load_cache() {
    this.engine = new webllm_cache.MLCEngine({
      appConfig: this.appConfig, // if do not specify, we use webllm.prebuiltAppConfig
      initProgressCallback: this.initProgressCallback,
    });
  }

  private load_origin() {
    this.originEngine = new webllm_origin.MLCEngine({
      appConfig: this.originAppConfig,
      initProgressCallback: this.initProgressCallback,
    });
  }

  async loadModel(model: string) {
    if (this.selectedLib === 1) {
      await this.originEngine.unload();
      this.selectedModel = this.originModel;
      await this.originEngine.reload(this.originModel);
    } else {
      this.engine.unload();
      this.selectedModel = model;
      await this.engine.reload(this.selectedModel);
    }
    this.isModelLoaded = true;
  }

  async run(
    input: string = "List three US states.",
    max_new_tokens: number = 4
  ) {
    if (!this.isModelLoaded) {
      setLabel("init-label", 'Models should be loaded before any submission.');
      return null;
    }

    if (this.selectedLib === 1) {
      return this.runOrigin(input, max_new_tokens);
    } else {
      return this.runWeInfer(input, max_new_tokens);
    }
  }

  private async runOrigin(input: string, max_new_tokens: number) {
    let timeRecord: any[] = [];
    let replyTokenCnt: number = 0;
    let fullReply = "";

    const genConfig = {
      top_p: 1.0,
      temperature: 0,
      presence_penalty: 0,
      frequency_penalty: 0,
      max_tokens: max_new_tokens,
    };

    const chunks = await this.originEngine.chat.completions.create({
      messages: [{ role: "user", content: input }],
      stream: true,
      stream_options: { include_usage: true },
      ...genConfig,
    });

    let lastStart = performance.now();
    let isFirstChunk = true;

    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        const now = performance.now();
        replyTokenCnt += 1;
        const tdelta = now - lastStart;

        if (isFirstChunk) {
          // First token includes prefill time
          timeRecord.push({ step: 1, message: delta, time: tdelta });
          isFirstChunk = false;
        } else {
          timeRecord.push({ step: replyTokenCnt, message: delta, time: tdelta });
        }
        console.log(`Time for #token ${replyTokenCnt}: ${tdelta} ms`);
        lastStart = now;
        fullReply += delta;
        setLabel("generate-label", fullReply);
      }
    }

    console.log(fullReply);
    return {
      reply: fullReply,
      replyTokenCnt,
      timeRecord,
      genConfig,
    };
  }

  private async runWeInfer(input: string, max_new_tokens: number) {
    let timeRecord: any[] = [];
    let lastStart: number = -1;
    let replyTokenCnt: number = 0;
    const generateProgressCallback = (_step: number, message: string) => {
      console.timeEnd(`token ${_step}`);
      const tend = performance.now();
      replyTokenCnt += 1;
      let tdelta = -1;
      if (lastStart !== -1) {
        tdelta = tend - lastStart;
        lastStart = tend;
        if (_step === 2) {
          const prefillTime = this.engine.getPipeline().prefillTotalTime * 1e3;
          timeRecord.push({ step: 1, message: message, time: prefillTime });
          tdelta -= prefillTime;
        }
        console.log(`Time for #token: ${_step}: ${tdelta} ms`);
        timeRecord.push({ step: _step, message: message, time: tdelta });
      }
      if (_step <= max_new_tokens) {
        console.time(`token ${_step + 1}`);
      }
      setLabel("generate-label", message);
    };
    let genConfig: webllm.GenerationConfig = {
      top_p: 1.0,
      temperature: 0,
      presence_penalty: 0,
      frequency_penalty: 0,
      max_tokens: max_new_tokens,
    }

    console.time(`token 2`);
    lastStart = performance.now();
    const reply = await this.engine.generate(input, generateProgressCallback, 1, genConfig);
    console.log(reply);
    return {
      reply,
      replyTokenCnt,
      timeRecord,
      genConfig,
    };
  }
}

function getCurrentModelName() {
  const selectElement = document.querySelector('#model-select');
  const index = selectElement.selectedIndex;
  const options = selectElement.options;
  const selectedModel = options[index].value;
  return selectedModel;
}

async function main() {
  const engine = new Engine();
  document.querySelector('#input').value = "Please introduce the Peking University in detail to me.";
  document.querySelector('#maxNew').value = 32;
  document.querySelector('#load').addEventListener('click', async () => {
    const selectedModel = getCurrentModelName();
    setLabel("init-label", 'Start to load model');
    await engine.loadModel(selectedModel);
  })

  async function runTask(engine, userInputText, maxNewTokens) {
    const tstart = performance.now();
    const result = await engine.run(userInputText, maxNewTokens);
    const tend = performance.now();
    const TotalTime = tend - tstart;
    if (result) {
      const TotalTokens = result.replyTokenCnt + 1; // prefill不会调回调
      const PrefillSpeed = (result.timeRecord[0].time);
      const DecodeSpeed = (TotalTime - PrefillSpeed) / (TotalTokens - 1);
      const AverageSpeed = (TotalTime) / (TotalTokens);
      const stats = {
        TotalTime,
        TotalTokens,
        PrefillSpeed,
        DecodeSpeed,
        AverageSpeed,
        inputText: userInputText,
        outputText: result.reply,
        config: {
          model: engine.selectedModel,
          lib: engine.selectedLib,
          x: engine.x,
          gpuLabel: engine.selectedLib === 1 ? 'Unknown' : (engine.engine?.gpuLabel ?? 'Unknown'),
          genConfig: result.genConfig,
        },
        TimeRecord: result.timeRecord,
      }
      console.log(stats);
      // sendResultToServer(stats, `${engine.selectedModel}`);
      setLabel('stats-label', `Decode Speed: ${DecodeSpeed} ms/token`);
      return stats;
    }
    return null;
  }

  document.querySelector('#submit')?.addEventListener('click', async () => {
    const userInputText = document.querySelector('#input')?.value;
    const maxNewTokens = +document.querySelector('#maxNew')?.value;
    if (userInputText) {
      console.log(`#max_token: ${maxNewTokens}\nInput:\n${userInputText}`);
      console.time(`Total Time for #token: ${maxNewTokens}`);
      await runTask(engine, userInputText, maxNewTokens);
      console.timeEnd(`Total Time for #token: ${maxNewTokens}`);
    }
  })

  const options = document.getElementsByName('options');
  options[2].checked = true;
  options.forEach(option => {
    option.addEventListener('change', () => {
      for (let option of options) {
        if (option.checked) {
          engine.loadEngine(+option.value);
          break;
        }
      }
    });
  });
  document.querySelector('#x_value').value = 4;
  document.querySelector('#x_value').addEventListener('change', () => {
    const x = +document.querySelector('#x_value').value;
    if (x > 0) {
      engine.updateX(x);
      console.log(`Change submit interval to ${x}`);
    }
  });
  setLabel("init-label", '');
}

setLabel("init-label", 'Loading...');
main();
