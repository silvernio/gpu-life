export function hslToRgb(
  h: number,
  s: number,
  l: number,
): [number, number, number] {
  h = ((h % 360) + 360) % 360;
  s = Math.max(0, Math.min(1, s));
  l = Math.max(0, Math.min(1, l));

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0,
    g = 0,
    b = 0;

  if (h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x;
  }

  return [r + m, g + m, b + m];
}

export async function logBufferf32(
  device: GPUDevice,
  buffer: GPUBuffer,
  size: number,
) {
  const readbackBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, size);

  device.queue.submit([commandEncoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);

  const arrayBuffer = readbackBuffer.getMappedRange();

  const outputData = new Float32Array(arrayBuffer);

  console.log(outputData.slice());

  readbackBuffer.unmap();
  readbackBuffer.destroy();
}

export async function logBufferu32(
  device: GPUDevice,
  buffer: GPUBuffer,
  size: number,
) {
  const readbackBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, size);

  device.queue.submit([commandEncoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);

  const arrayBuffer = readbackBuffer.getMappedRange();

  const outputData = new Uint32Array(arrayBuffer);

  console.log(outputData.slice());

  readbackBuffer.unmap();
  readbackBuffer.destroy();
}

let canTimestamp = false;
let canSubgroups = false;
const timestamps: Record<
  string,
  {
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
    v: number;
  }
> = {};

export async function requestTimestamps(adapter: GPUAdapter) {
  canTimestamp = adapter.features.has('timestamp-query');
  canSubgroups = adapter.features.has('subgroups');

  const urlParams = new URLSearchParams(window.location.search);
  const flag = urlParams.has('noTimestamp');
  if (flag) {
    canTimestamp = false;
  }

  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(canTimestamp ? ['timestamp-query' as GPUFeatureName] : []),
      ...(canSubgroups ? ['subgroups' as GPUFeatureName] : [])
    ]
  })

  if (!device.features.has('timestamp-query')) {
    canTimestamp = false;
  }
  if (!device.features.has('subgroups')) {
    canSubgroups = false;
  }

  return device;

}

export function setupTimestamp(device: GPUDevice, name: string) {
  if (!canTimestamp) return;
  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: 2,
  });
  timestamps[name] = {
    querySet,
    resolveBuffer: device.createBuffer({
      size: 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    resultBuffer: device.createBuffer({
      size: 2 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    v: 0,
  };
}

export function linkComputeTimestamp(
  device: GPUDevice,
  name: string,
): GPUComputePassDescriptor {
  device.pushErrorScope('validation');
  if (!canTimestamp) {
    return {};
  }
  return {
    timestampWrites: {
      querySet: timestamps[name].querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  };
}

export function linkRenderTimestamp(
  device: GPUDevice,
  description: GPURenderPassDescriptor,
  name: string,
): GPURenderPassDescriptor {
  device.pushErrorScope('validation');
  if (!canTimestamp) {
    return description;
  }
  return {
    ...description,
    timestampWrites: {
      querySet: timestamps[name].querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  };
}

export function resolveTimestamp(
  commandEncoder: GPUCommandEncoder,
  name: string,
) {
  if (!canTimestamp) return;
  commandEncoder.resolveQuerySet(
    timestamps[name].querySet,
    0,
    timestamps[name].querySet.count,
    timestamps[name].resolveBuffer,
    0,
  );
  if (timestamps[name].resultBuffer.mapState == 'unmapped') {
    commandEncoder.copyBufferToBuffer(
      timestamps[name].resolveBuffer,
      0,
      timestamps[name].resultBuffer,
      0,
      timestamps[name].resultBuffer.size,
    );
  }
}

export async function readTimestamp(name: string) {
  if (!canTimestamp) return 0;
  if (!canTimestamp || timestamps[name].resultBuffer.mapState != 'unmapped')
    return timestamps[name].v;

  await timestamps[name].resultBuffer.mapAsync(GPUMapMode.READ);

  const times = new BigUint64Array(
    timestamps[name].resultBuffer.getMappedRange(),
  );

  timestamps[name].v = Number(times[1] - times[0]) / 1000 / 1000; // ms
  timestamps[name].resultBuffer.unmap();

  return timestamps[name].v;
}

export function lerpn(
  start: number,
  end: number,
  multiply: number,
  step: number,
) {
  multiply = 1 - (1 - multiply) ** step;
  if (multiply > 1) multiply = 1;
  if (multiply < 0) multiply = 0;
  return start + (end - start) * multiply;
}

export function lerp5(start: number, end: number, step: number) {
  return lerpn(start, end, 0.5, step);
}
