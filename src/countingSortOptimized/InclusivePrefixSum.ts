import prefixSumCommonsWGSL from './computeShaders/commons.wgsl?raw';
import { ReduceCompute } from './computeShaders/reduce';
import { SpineScanShortCompute } from './computeShaders/spineScanShort';
import { SpineScanLongCompute } from './computeShaders/spineScanLong';
import { DownSweepCompute } from './computeShaders/downsweep';
import { linkComputeTimestamp, readTimestamp, resolveTimestamp, setupTimestamp } from '../utils';
import { directivesCode } from './computeShaders/directives';

type PrefixSumElementType = 'float' | 'int' | 'uint';
type PrefixSumVecType = 'vec4' | 'ivec4' | 'uvec4';
// type SupportedTypedArray = Float32Array | Int32Array | Uint32Array;
type PrefixSumPipelineName =
  | 'reduce'
  | 'spineScanShort'
  | 'spineScanLong'
  | 'downSweep';

interface PrefixSumOptions {
  workPerInvocation?: number;
  workgroupSize?: number;
}

interface PipelineManifestInterface {
  name: PrefixSumPipelineName
  code: string
  layouts: GPUBindGroupLayout[]
  computeConstants?: Record<string, number>
}

const divRoundUp = (size: number, part_size: number): number => {

  return Math.floor((size + part_size - 1) / part_size);
};

/**
 * A class that represents an inclusive prefix sum running under the reduce/scan strategy.
 * Currently limited to one-dimensional data buffers.
 *
 * @param {GPUDevice} device - A renderer with the ability to execute compute operations.
 * @param {TypedArray} inputArray - The data buffer to sum.
 * @param {Object} [options={}] - Options that modify the reduce/scan prefix sum.
 */
export class InclusivePrefixSum {
  device: GPUDevice;
  type: PrefixSumElementType;
  vecType: PrefixSumVecType;
  count: number;
  vecCount: number;
  unvectorizedWorkPerInvocation: number;
  workgroupSize: number;
  minSubgroupSize: number;
  partitionSize: number;
  numWorkgroups: number;
  dispatchSize: number;
  pipelines: Record<PrefixSumPipelineName, GPUComputePipeline>;
  // inputArrayBuffer: SupportedTypedArray;
  workPerInvocation: number;

  // Internal: one u32 partial-reduction result per workgroup
  reductionBuffer: GPUBuffer;
  // Uniform: PrefixSumParams { workgroupCount, workgroupSize, vecCount, workPerInvocation }
  paramsBuffer: GPUBuffer;

  // @group(0): binding 0 = input (Vec4), binding 1 = reduction (u32), binding 2 = output (u32)
  dataBindGroupLayout: GPUBindGroupLayout;
  dataBindGroup: GPUBindGroup;

  // @group(1): binding 0 = params uniform
  paramsBindGroupLayout: GPUBindGroupLayout;
  paramsBindGroup: GPUBindGroup;

  /**
   * Constructs a new PrefixSum helper.
   *
   * @param {GPUDevice} device - A renderer with the ability to execute compute operations.
   * @param {boolean} linearIndexingAvailable - Flag indicating whether the current device can use the WGSL linear_indexing builtin.
   * @param {GPUBuffer} inputVecBuffer - The input buffer that reads the initial counts (gets interpreted as a vec4)
   * @param {GPUBuffer} outputBuffer - The output buffer the inclusive prefix sum is written into.
   * @param {number} outputBufferOffset - The offset into the output buffer to write to. Useful if you need to represent the output as an exclusive prefix sum (offset of 1) or if the counts get summed again.
   * @param {Object} [options={}] - Options that modify the behavior of the prefix sum.
   */
  constructor(
    device: GPUDevice,
    inputVecBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    outputBufferOffset: number,
    // inputArray: SupportedTypedArray,
    options: PrefixSumOptions = {}
  ) {
    this.device = device;
    /**
     * The type of each individual data element.
     *
     * @type {string}
     */
    this.type = 'uint';

    this.vecType = 'uvec4';

    /**
     * The size of the data.
     *
     * @type {number}
     */
    this.count = inputVecBuffer.size / Uint32Array.BYTES_PER_ELEMENT;

    /**
     * The number of 4-dimensional vectors needed to fully represent the data in the data buffer.
     * Buffers where this.count % 4 !== 0 will need an additional vec4 to hold the data buffer's
     * remaining elements.
     *
     * @type {number}
     */
    this.vecCount = divRoundUp(this.count, 4);

    /**
     * The number of 4-dimensional vectors that will be read from global storage in each invocation of the reduction/downsweep step.
     * Defaults to 4.
     *
     * @type {number}
     */
    this.workPerInvocation = options.workPerInvocation
      ? options.workPerInvocation
      : 4;

    /**
     * The number of unvectorized values to be read from the reduction buffer in each invocation of the spine/scan step.
     * Derived from workPerInvocation and thus defaults to 16.
     *
     * @type {number}
     */
    this.unvectorizedWorkPerInvocation = this.workPerInvocation * 4;

    /**
     * The workgroup size of the compute shaders executed during the prefix sum.
     * If no workgroupSize is defined, the workgroupSize defaults to the minimumn between the number of elements in the
     * data buffer and 64.
     *
     * @type {number}
     */
    this.workgroupSize = options.workgroupSize
      ? options.workgroupSize
      : Math.min(this.vecCount, this.device.limits.maxComputeWorkgroupSizeX);

    /**
     * The minimumn subgroup size specified by the renderer's graphics device.
     *
     * @type {number}
     */
    this.minSubgroupSize =
      this.device.adapterInfo && this.device.adapterInfo.subgroupMinSize
        ? this.device.adapterInfo.subgroupMinSize
        : 4;

    /**
     * The maximum number of elements that will be read by an individual workgroup in the reduction step.
     * Calculated as the number of invocations in the workgroup by the work per invocation by VEC4_SIZE
     *
     * @type {number}
     */
    this.partitionSize =
      this.workgroupSize * this.unvectorizedWorkPerInvocation;

    /**
     * The number of workgroups needed to properly execute the reduction and downsweepsteps.
     * Calculated as the number of partitions within the count of elements.
     *
     * @type {number}
     */
    this.numWorkgroups = divRoundUp(this.count, this.partitionSize);

    /**
     * The number of invocations dispatched in each step of the prefix sum.
     *
     * @type {number}
     */
    this.dispatchSize = this.numWorkgroups * this.workgroupSize;

    // Internal reduction buffer: one u32 per workgroup
    this.reductionBuffer = device.createBuffer({
      label: 'PrefixSum.reductionBuffer',
      size: Math.max(this.numWorkgroups, 1) * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Uniform: PrefixSumParams { workgroupCount, workgroupSize, vecCount, workPerInvocation }
    this.paramsBuffer = device.createBuffer({
      label: 'PrefixSum.paramsBuffer',
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([
        this.numWorkgroups,
        this.workgroupSize,
        this.vecCount,
        this.workPerInvocation,
      ])
    );

    // @group(0): binding 0 = input (Vec4), binding 1 = reduction (u32), binding 2 = output (u32)
    this.dataBindGroupLayout = device.createBindGroupLayout({
      label: 'PrefixSum.dataBindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });

    // @group(1): binding 0 = params uniform
    this.paramsBindGroupLayout = device.createBindGroupLayout({
      label: 'PrefixSum.paramsBindGroupLayout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
      ],
    });

    this.pipelines = {} as Record<PrefixSumPipelineName, GPUComputePipeline>;

    const hasLinearIndexing = navigator.gpu.wgslLanguageFeatures.has('linear-indexing');
    const hasSubgroupID = navigator.gpu.wgslLanguageFeatures.has('subgroup_id')

    // spineScanShort only uses @group(0); all others also use @group(1) for params
    const prefixSumPipelinesManifest: PipelineManifestInterface[] = [
      {
        name: 'reduce',
        code: directivesCode(hasLinearIndexing, false) + prefixSumCommonsWGSL + ReduceCompute(
          this.workgroupSize,
          hasLinearIndexing,
          false
        ),
        layouts: [this.dataBindGroupLayout, this.paramsBindGroupLayout],
      },
      {
        name: 'spineScanShort',
        code: directivesCode(hasLinearIndexing, false) + prefixSumCommonsWGSL + SpineScanShortCompute(
          this.workgroupSize,
          hasLinearIndexing
        ),
        layouts: [this.dataBindGroupLayout],
      },
      {
        name: 'spineScanLong',
        code: directivesCode(hasLinearIndexing, false) + prefixSumCommonsWGSL + SpineScanLongCompute(
          this.workgroupSize,
          hasLinearIndexing,
          false
        ),
        layouts: [this.dataBindGroupLayout, this.paramsBindGroupLayout],
      },
      {
        name: 'downSweep',
        code:
          directivesCode(hasLinearIndexing, false) +
          prefixSumCommonsWGSL +
          DownSweepCompute(this.workgroupSize, hasLinearIndexing, false),
        layouts: [this.dataBindGroupLayout, this.paramsBindGroupLayout],
        computeConstants: {
          OUTPUT_INDEX_OFFSET: outputBufferOffset,
        },
      },
    ];


    for (const manifest of prefixSumPipelinesManifest) {
      const computeProgram: GPUProgrammableStage = {
        module: this.device.createShaderModule({
          code: manifest.code,
        }),
      };

      if (manifest.computeConstants) {
        computeProgram.constants = manifest.computeConstants;
      }

      this.pipelines[manifest.name] = device.createComputePipeline({
        label: `computePipeline.prefixSum_${manifest.name}`,
        layout: device.createPipelineLayout({
          bindGroupLayouts: manifest.layouts,
        }),
        compute: computeProgram,
      });

      setupTimestamp(device, manifest.name);
    }

    this.dataBindGroup = device.createBindGroup({
      label: 'PrefixSum.dataBindGroup',
      layout: this.dataBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputVecBuffer } },
        { binding: 1, resource: { buffer: this.reductionBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });

    this.paramsBindGroup = device.createBindGroup({
      label: 'PrefixSum.paramsBindGroup',
      layout: this.paramsBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.paramsBuffer } }],
    });
  }

  run(commandEncoder: GPUCommandEncoder) {
    const reducePass = commandEncoder.beginComputePass(
      linkComputeTimestamp(this.device, 'reduce')
    );
    reducePass.setPipeline(this.pipelines['reduce']);
    reducePass.setBindGroup(0, this.dataBindGroup);
    reducePass.setBindGroup(1, this.paramsBindGroup);
    reducePass.dispatchWorkgroups(this.numWorkgroups);
    reducePass.end();

    if (this.numWorkgroups <= this.minSubgroupSize) {
      const spineScanPass = commandEncoder.beginComputePass(
        linkComputeTimestamp(this.device, 'spineScanShort')
      );
      spineScanPass.setBindGroup(0, this.dataBindGroup);
      spineScanPass.setPipeline(this.pipelines['spineScanShort']);
      spineScanPass.dispatchWorkgroups(1);
      spineScanPass.end();
      resolveTimestamp(commandEncoder, 'spineScanShort');
    } else {
      const spineScanPass = commandEncoder.beginComputePass(
        linkComputeTimestamp(this.device, 'spineScanLong')
      );
      spineScanPass.setBindGroup(0, this.dataBindGroup);
      spineScanPass.setPipeline(this.pipelines['spineScanLong']);
      spineScanPass.setBindGroup(1, this.paramsBindGroup);
      spineScanPass.dispatchWorkgroups(1);
      spineScanPass.end();
      resolveTimestamp(commandEncoder, 'spineScanLong');
    }

    const downSweepPass = commandEncoder.beginComputePass(
      linkComputeTimestamp(this.device, 'downSweep')
    );
    downSweepPass.setPipeline(this.pipelines['downSweep']);
    downSweepPass.setBindGroup(0, this.dataBindGroup);
    downSweepPass.setBindGroup(1, this.paramsBindGroup);
    downSweepPass.dispatchWorkgroups(this.numWorkgroups);
    downSweepPass.end();
    resolveTimestamp(commandEncoder, 'downSweep')
  }


  updateDisplays(params: Record<string, number>) {
    const spineScanName = this.numWorkgroups <= this.minSubgroupSize ? 'spineScanShort' : 'spineScanLong';
    Promise.all([
      readTimestamp('reduce'),
      readTimestamp(spineScanName),
      readTimestamp('downSweep'),
    ]).then(([reduceTime, spineScanTime, downSweepTime]) => {
      params.prefix = reduceTime + spineScanTime + downSweepTime;
    });
  }

}
