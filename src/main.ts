import { tcamera, mouse } from './input';
import renderShaders from './render.wgsl?raw';
import {
  hslToRgb,
  lerp5,
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';
import {
  engineSelect,
  globalPerformanceParams,
  newSimBtn,
  optionParams,
  params,
  performanceParams,
  randomizeBtn,
  setEngineDisplay,
  setSim,
  particleAmt as particleAmtOptn,
  colourAmt as colourAmtOptn,
  bindOptions,
} from './options';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

const uniformsSize = 16;
const uniformData = new Float32Array(uniformsSize);

const simSize = 16;
const simData = new Float32Array(simSize);

function makeRandomMatrix() {
  const rows = [];
  for (let i = 0; i < colourAmt; i++) {
    const row = [];
    for (let j = 0; j < colourAmt; j++) {
      row.push(Math.random() * 2 - 1);
    }
    rows.push(row);
  }
  return rows;
}

let colourAmt = colourAmtOptn;
let colours: [number, number, number][] = [];
for (let i = 0; i < colourAmt; i++) {
  colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
}
let matrix = makeRandomMatrix();

const particleStride = 24;

const multistep = 1;

const camera = { ...tcamera };
const cameraData = new Float32Array([camera.x, camera.y, camera.zoom]);

let particleAmt = particleAmtOptn;

let device: GPUDevice | undefined;
let context: GPUCanvasContext | undefined;
let uniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let renderPipeline: GPURenderPipeline | undefined;
let cameraBuffer: GPUBuffer | undefined;

let matrixBuffer: GPUBuffer | undefined;
let colourBuffer: GPUBuffer | undefined;

let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;

let renderBindGroup: GPUBindGroup | undefined;

let alternate = 0;
let fpsc = 0;

import * as nSquared from './nSquared/main';
import * as linkedList from './linkedList/main';
import * as countingSort from './countingSort/main';
import * as prefixSum from './countingSortOptimized/main';
const engines: Record<
  string,
  typeof nSquared | typeof linkedList | typeof countingSort | typeof prefixSum
> = {
  nSquared,
  linkedList,
  countingSort,
};

let engine: string = 'countingSort';
setEngineDisplay(engine);

(async () => {
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });

  if (!adapter) return;

  device = await requestTimestamps(adapter);

  if (device?.features.has('subgroups')) {
    engines['prefixSum'] = prefixSum;
  }


  context = canvas.getContext('webgpu') ?? undefined;

  if (!context) return;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({ device, format: presentationFormat });

  //

  uniformBuffer = device.createBuffer({
    size: uniformsSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'uniformBuffer',
  });

  simBuffer = device.createBuffer({
    size: simSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'simBuffer',
  });

  cameraBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'cameraBuffer',
  });

  //

  for (const engine in engines) {
    engines[engine].setup(device);
  }

  const renderModule = device.createShaderModule({ code: renderShaders });

  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vertex',
      buffers: [
        {
          arrayStride: particleStride,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x2' },
            { shaderLocation: 2, offset: 16, format: 'float32' },
          ],
        },
      ],
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fragment',
      targets: [
        {
          format: presentationFormat,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add',
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'zero',
              operation: 'add',
            },
          },
        },
      ],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  setupTimestamp(device, 'render');

  startParticles();

  bindOptions(device, simData, simBuffer);
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device) return;
  engines[engine].tick(device, commandEncoder, alternate, particleAmt);
  alternate = (alternate + 1) % 2;
}

function render(context: GPUCanvasContext, commandEncoder: GPUCommandEncoder) {
  if (!renderPipeline || !particleBuffers || !device) return;

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  const passEncoder = commandEncoder.beginRenderPass(
    linkRenderTimestamp(device, renderPassDescriptor, 'render'),
  );
  passEncoder.setPipeline(renderPipeline);
  passEncoder.setVertexBuffer(0, particleBuffers[(alternate + 1) % 2]);
  passEncoder.setBindGroup(0, renderBindGroup);
  passEncoder.draw(6, particleAmt, 0, 0);
  passEncoder.end();

  resolveTimestamp(commandEncoder, 'render');
}

function startParticles() {
  if (
    !device ||
    !uniformBuffer ||
    !renderPipeline ||
    !simBuffer ||
    !cameraBuffer
  )
    return;

  const bufferSize = particleAmt * particleStride;
  particleBuffers = [
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST,
    }),
  ];

  alternate = 0;
  const data = new Float32Array(bufferSize / 4);
  let pi = 0;
  while (pi < particleAmt) {
    const spawnAmt = ((Math.random() * (particleAmt - pi)) / colourAmt) * 5;
    const c = Math.floor(Math.random() * colourAmt);

    const a = Math.random() * Math.PI * 2;
    const d = Math.random() * optionParams.worldSize * 0.9;

    const x = Math.cos(a) * d;
    const y = Math.sin(a) * d;
    for (let i = 0; i < spawnAmt; i++) {
      const a = Math.random() * Math.PI * 2;
      const d = (Math.random() ** 3 / 10) * optionParams.worldSize;
      data[pi * 6] = x + Math.cos(a) * d;
      data[pi * 6 + 1] = y + Math.sin(a) * d;
      data[pi * 6 + 2] = 0;
      data[pi * 6 + 3] = 0;
      data[pi * 6 + 4] = c;

      data[pi * 6 + 5] = 0;

      pi++;
    }
  }

  device.queue.writeBuffer(particleBuffers[0], 0, data.buffer);

  setSim(device, simData, simBuffer);

  matrixBuffer = device.createBuffer({
    size: colourAmt * colourAmt * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const matrixData = new Float32Array(colourAmt * colourAmt);
  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      matrixData[c1 * colourAmt + c2] = matrix[c1][c2];
    }
  }
  device.queue.writeBuffer(matrixBuffer, 0, matrixData.buffer);

  colourBuffer = device.createBuffer({
    size: colourAmt * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const colourData = new Float32Array(colourAmt * 3);
  for (let c = 0; c < colourAmt; c++) {
    colourData[c * 3] = colours[c][0];
    colourData[c * 3 + 1] = colours[c][1];
    colourData[c * 3 + 2] = colours[c][2];
  }
  device.queue.writeBuffer(colourBuffer, 0, colourData.buffer);

  for (const engine in engines) {
    engines[engine].start(
      device,
      uniformBuffer,
      simBuffer,
      matrixBuffer,
      particleBuffers,
      particleAmt,
      params.cells,
    );
  }

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: cameraBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: colourBuffer,
        },
      },
    ],
  });
}

let lastTime = 0;
const deltaVs: number[] = [];

function update() {
  requestAnimationFrame(update);
  if (!device || !context) return;

  const start = performance.now();
  const delta = (start - lastTime) / 1000;
  deltaVs.push(start - lastTime);
  if (deltaVs.length > 1000) {
    deltaVs.splice(0, 1);
  }
  lastTime = start;

  camera.x = lerp5(camera.x, tcamera.x, delta * 50);
  camera.y = lerp5(camera.y, tcamera.y, delta * 50);
  camera.zoom = lerp5(camera.zoom, tcamera.zoom, delta * 50);

  const commandEncoder = device.createCommandEncoder();

  if (uniformBuffer) {
    uniformData[0] = canvas.width / canvas.height;

    uniformData[4] = mouse.x;
    uniformData[5] = mouse.y;
    uniformData[6] = mouse.down;
    uniformData[7] = mouse.type;

    uniformData[8] = (1 / optionParams.r) * 0.015;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  }

  if (cameraBuffer) {
    cameraData[0] = camera.x;
    cameraData[1] = camera.y;
    cameraData[2] = camera.zoom;

    device.queue.writeBuffer(cameraBuffer, 0, cameraData);
  }

  for (let i = 0; i < multistep; i++) {
    tick(commandEncoder);
  }

  render(context, commandEncoder);

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  device.popErrorScope().then((error) => {
    if (error) {
      // some weird bug happened with timestamps, just disable it and restart the simulation
      window.location.href +=
        (window.location.search ? '&' : '?') + 'noTimestamp';
    }
  });

  const cpuTime = performance.now() - start;
  globalPerformanceParams.cpu = cpuTime;
  updateTotal();

  for (const engine2 in engines) {
    if (engine == engine2) {
      engines[engine2].updateDisplays(
        performanceParams[engine2 as keyof typeof performanceParams],
      );
    }
  }

  readTimestamp('render').then((time) => {
    globalPerformanceParams.render = time;
    updateTotal();
  });

  fpsc++;
}

requestAnimationFrame(update);

newSimBtn.on('click', () => {
  colourAmt = optionParams.colours;
  colours = [];
  for (let i = 0; i < colourAmt; i++) {
    colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
  }

  params.cells = optionParams.cells;
  particleAmt = params.particles;

  matrix = makeRandomMatrix();
  startParticles();
});

randomizeBtn.on('click', () => {
  if (!device || !matrixBuffer) return;
  matrix = makeRandomMatrix();
  const matrixData = new Float32Array(colourAmt * colourAmt);
  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      matrixData[c1 * colourAmt + c2] = matrix[c1][c2];
    }
  }
  device.queue.writeBuffer(matrixBuffer, 0, matrixData.buffer);
});

setInterval(() => {
  params.fps = fpsc;
  fpsc = 0;
}, 1000);

engineSelect.on('change', (event) => {
  engine = event.value;
  setEngineDisplay(engine);
});

function updateTotal() {
  let deltaTotal = 0;
  for (const time of deltaVs) {
    deltaTotal += time;
  }
  deltaTotal /= deltaVs.length;
  if (isNaN(deltaTotal)) deltaTotal = 0;

  let total = 0;
  const params = performanceParams[engine as keyof typeof performanceParams];
  for (const pass in params) {
    total += params[pass as keyof typeof params];
  }
  total += globalPerformanceParams.cpu;
  total += globalPerformanceParams.render;
  globalPerformanceParams.total = total;
  globalPerformanceParams.graph = total / deltaTotal;
}

window.onresize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
};
