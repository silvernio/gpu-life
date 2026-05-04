import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import { cellComputeShader } from './computeShaders/cell';
import { sortComputeShader } from './computeShaders/sort';
import { simComputeShader } from './computeShaders/sim';
import { InclusivePrefixSum } from './InclusivePrefixSum';

let workgroupSize = 128;

let cellPipeline: GPUComputePipeline | undefined;
let sortPipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;

let cellBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;
let sortBindGroup: GPUBindGroup | undefined;
let simBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;

let cellBuffer: GPUBuffer | undefined;
let countBuffer: GPUBuffer | undefined;
let prefixSumBuffer: GPUBuffer | undefined;
let zeroBuffer: GPUBuffer | undefined;
let sortedBuffer: GPUBuffer | undefined;

let device: GPUDevice | undefined;

let prefixSumModule: InclusivePrefixSum | undefined;

export function setup(device2: GPUDevice) {
  device = device2;

  workgroupSize = device.limits.maxComputeWorkgroupSizeX;

  const cellModule = device.createShaderModule({
    code: cellComputeShader(workgroupSize),
  });

  setupTimestamp(device, 'cell');

  cellPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: cellModule,
      entryPoint: 'main',
    },
  });

  //

  const sortModule = device.createShaderModule({
    code: sortComputeShader(workgroupSize),
  });

  setupTimestamp(device, 'sort');

  sortPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: sortModule,
      entryPoint: 'main',
    },
  });

  //

  const simModule = device.createShaderModule({
    code: simComputeShader(workgroupSize),
  });

  setupTimestamp(device, 'countSim');

  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: simModule,
      entryPoint: 'main',
    },
  });
}

export function start(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  simBuffer: GPUBuffer,
  matrixBuffer: GPUBuffer,
  particleBuffers: [GPUBuffer, GPUBuffer],
  particleAmt: number,
  cellAmt: number,
) {
  if (!cellPipeline || !sortPipeline || !simPipeline) return;

  cellBuffer = device.createBuffer({
    size:
      particleAmt *
      (8 * Float32Array.BYTES_PER_ELEMENT + 1 * Uint32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'cellBuffer',
  });
  sortedBuffer = device.createBuffer({
    size:
      particleAmt *
      (8 * Float32Array.BYTES_PER_ELEMENT + 1 * Uint32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'sortedBuffer',
  });
  countBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    label: 'countBuffer',
  });

  prefixSumBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT + (2 * Uint32Array.BYTES_PER_ELEMENT),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'prefixSumBuffer'
  })

  zeroBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'zeroBuffer',
  });

  device.queue.writeBuffer(zeroBuffer, 0, new Uint32Array(cellAmt));

  const cellGroups = [];
  for (let i = 0; i < 2; i++) {
    cellGroups.push(
      device.createBindGroup({
        layout: cellPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: simBuffer,
            },
          },
          {
            binding: 1,
            resource: {
              buffer: particleBuffers[i],
            },
          },
          {
            binding: 2,
            resource: {
              buffer: cellBuffer,
            },
          },
          {
            binding: 3,
            resource: {
              buffer: countBuffer,
            },
          },
        ],
      }),
    );
  }
  cellBindGroups = [cellGroups[0], cellGroups[1]];

  sortBindGroup = device.createBindGroup({
    layout: sortPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: cellBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: sortedBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: prefixSumBuffer,
        },
      },
    ],
  });

  const simGroups = [];
  for (let i = 0; i < 2; i++) {
    simGroups.push(
      device.createBindGroup({
        layout: simPipeline.getBindGroupLayout(0),
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
              buffer: simBuffer,
            },
          },
          {
            binding: 2,
            resource: {
              buffer: matrixBuffer,
            },
          },
          {
            binding: 3,
            resource: {
              buffer: particleBuffers[i],
            },
          },
          {
            binding: 4,
            resource: {
              buffer: particleBuffers[1 - i],
            },
          },
          {
            binding: 5,
            resource: {
              buffer: sortedBuffer,
            },
          },
          {
            binding: 6,
            resource: {
              buffer: countBuffer
            }
          },
          {
            binding: 7,
            resource: {
              buffer: prefixSumBuffer,
            },
          },
        ],
      }),
    );
  }
  simBindGroups = [simGroups[0], simGroups[1]];

  prefixSumModule = new InclusivePrefixSum(
    device,
    navigator.gpu.wgslLanguageFeatures.has('linear-indexing'),
    countBuffer,
    prefixSumBuffer,
    2
  );
}

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
) {
  if (
    !cellPipeline ||
    !cellBindGroups ||
    !zeroBuffer ||
    !countBuffer ||
    !sortPipeline ||
    !sortBindGroup ||
    !simPipeline ||
    !simBindGroups ||
    !prefixSumModule
  )
    return;

  commandEncoder.copyBufferToBuffer(zeroBuffer, 0, countBuffer, 0, countBuffer.size);

  const cellPassEncoder = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'cell'),
  );
  cellPassEncoder.setPipeline(cellPipeline);
  cellPassEncoder.setBindGroup(0, cellBindGroups[alternate]);
  cellPassEncoder.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  cellPassEncoder.end();
  resolveTimestamp(commandEncoder, 'cell');

  //

  prefixSumModule?.run(commandEncoder)

  const sortPassEncoder = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'sort'),
  );
  sortPassEncoder.setPipeline(sortPipeline);
  sortPassEncoder.setBindGroup(0, sortBindGroup);
  sortPassEncoder.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  sortPassEncoder.end();
  resolveTimestamp(commandEncoder, 'sort');

  //

  const simPassEncoder = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'countSim'),
  );
  simPassEncoder.setPipeline(simPipeline);
  simPassEncoder.setBindGroup(0, simBindGroups[alternate]);
  simPassEncoder.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  simPassEncoder.end();

  resolveTimestamp(commandEncoder, 'countSim');
}

export function updateDisplays(params: Record<string, number>) {
  if (!prefixSumModule) {
    return;
  }
  readTimestamp('cell').then((time) => {
    params.cell = time;
  });
  prefixSumModule?.updateDisplays(params);
  readTimestamp('sort').then((time) => {
    params.sort = time;
  });
  readTimestamp('countSim').then((time) => {
    params.sim = time;
  });
}
