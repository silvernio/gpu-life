import { constructBuiltinDeclarations } from './utils';

export const SpineScanShortCompute = (
  workgroupSize: number,
  linearIndexingAvailable: boolean
): string => {
  const builtinDeclarations = constructBuiltinDeclarations(
    linearIndexingAvailable
  );
  return /* wgsl */ `
// system
${!linearIndexingAvailable ? 'var<private> instanceIndex : u32;' : ''}

struct Prefix_Sum_Reduction_0Struct {
	value : array< u32 >
};

@group( 0 ) @binding( 1 )
var<storage, read_write> Prefix_Sum_Reduction_0 : Prefix_Sum_Reduction_0Struct;

@compute @workgroup_size( ${workgroupSize}, 1, 1 )
fn spineScanShort(
${builtinDeclarations}
) {

	// system
	${
    !linearIndexingAvailable &&
    `instanceIndex = globalId.x + globalId.y * ( ${workgroupSize} * numWorkgroups.x ) + globalId.z * ( ${workgroupSize} * numWorkgroups.x ) * ( 1 * numWorkgroups.y );`
  }

	Prefix_Sum_Reduction_0.value[ invocationSubgroupIndex ] = subgroupInclusiveAdd( Prefix_Sum_Reduction_0.value[ invocationSubgroupIndex ] );

}
`;
};
