import { constructBuiltinDeclarations } from './utils';

export const ReduceCompute = (
  workgroupSize: number,
  linearIndexingAvailable: boolean,
  subgroupIDAvailable: boolean
): string => {
  const builtinDeclarations = constructBuiltinDeclarations(
    linearIndexingAvailable,
	subgroupIDAvailable
  );
  return /* wgsl */ `

// system
${!linearIndexingAvailable ? 'var<private> instanceIndex : u32;' : ''}

// locals
var<workgroup> WorkgroupArray_898: array< u32, ${workgroupSize / 4} >;

@group( 0 ) @binding( 0 )
var<storage, read_write> Prefix_Sum_Input_Vec_0 : Vec4ArrayStruct;

@group( 0 ) @binding(1)
var<storage, read_write> Prefix_Sum_Reduction_0 : U32ArrayStruct;

@group(1) @binding(0) var<uniform> params: PrefixSumParams;

@compute @workgroup_size( ${workgroupSize}, 1, 1 )
fn reduce(
${builtinDeclarations}
) {

	${!linearIndexingAvailable ? `instanceIndex = globalId.x + globalId.y * ( ${workgroupSize} * numWorkgroups.x ) + globalId.z * ( ${workgroupSize} * numWorkgroups.x ) * ( 1 * numWorkgroups.y );` : ''}

	var subgroupOffset : u32;
	var threadSubgroupOffset : u32;
	var workgroupOffset : u32;
	var startThreadBase : u32;
	var startThread : u32;
	var subgroupReduction : u32;
	var subgroupSizeLog : u32;
	var spineSize : u32;
	var spineSizeLog : u32;
	var nodeVar2 : u32;
	var subgroupAlignedSize : u32;
	var workgroupSectionOffset : u32;
	var isValidSubgroupIndex : bool;
	var t : u32;

	${!subgroupIDAvailable ? `var invocationSubgroupMetaIndex: u32 = ( invocationLocalIndex / subgroupSize );` : ''}
	// subgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * 4u );
    subgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * params.workPerInvocation );
	threadSubgroupOffset = ( subgroupOffset + invocationSubgroupIndex );
	// workgroupOffset = ( workgroupId.x * ( 256u * 4u ) );
    workgroupOffset = ( workgroupId.x * ( params.workgroupSize * params.workPerInvocation ) );
	startThreadBase = ( threadSubgroupOffset + workgroupOffset );
	startThread = startThreadBase;
	subgroupReduction = 0u;

	// if ( ( workgroupId.x < ( 8u - 1u ) ) ) {
    if ( ( workgroupId.x < ( params.workgroupCount - 1u ) ) ) {

		// for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < 4u; currentSubgroupInBlock ++ ) {
        for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			subgroupReduction = ( subgroupReduction + u32( dot( vec4<u32>( 1u, 1u, 1u, 1u ), Prefix_Sum_Input_Vec_0.value[ startThread ] ) ) );
			startThread = ( startThread + subgroupSize );

		}

	}


	// if ( ( workgroupId.x == ( 8u - 1u ) ) ) {
    if ( ( workgroupId.x == ( params.workgroupCount - 1u ) ) ) {


		for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			subgroupReduction = ( subgroupReduction + u32( dot( select( vec4<u32>( 0u, 0u, 0u, 0u ), Prefix_Sum_Input_Vec_0.value[ startThread ], ( startThread < params.vecCount ) ), vec4<u32>( 1u, 1u, 1u, 1u ) ) ) );
			startThread = ( startThread + subgroupSize );

		}



	}

	subgroupReduction = subgroupAdd( subgroupReduction );

	if ( ( invocationSubgroupIndex == 0u ) ) {

		WorkgroupArray_898[ invocationSubgroupMetaIndex ] = subgroupReduction;


	}

	workgroupBarrier();
	subgroupSizeLog = countTrailingZeros( subgroupSize );
	// spineSize = ( 256u >> subgroupSizeLog );
    spineSize = ( params.workgroupSize >> subgroupSizeLog );
	spineSizeLog = countTrailingZeros( spineSize );
	nodeVar2 = ( ( spineSizeLog + subgroupSizeLog ) - 1u );
	nodeVar2 = ( nodeVar2 / subgroupSizeLog );
	nodeVar2 = ( nodeVar2 * subgroupSizeLog );
	subgroupAlignedSize = ( 1u << nodeVar2 );
	workgroupSectionOffset = 0u;

	// In cases where the number of subgroups in a workgroup is greater than the subgroup size itself,
	// we need to iterate over the array again to capture all the data in the workgroup array buffer
	// In many cases this loop will only run once
	for ( var j : u32 = subgroupSize; j <= subgroupAlignedSize; j <<= subgroupSizeLog ) {

		var subgroupIndex = ( ( ( invocationLocalIndex + 1u ) << workgroupSectionOffset ) - 1u );
		isValidSubgroupIndex = ( subgroupIndex < spineSize );
		t = subgroupAdd( select( 0u, WorkgroupArray_898[ subgroupIndex ], isValidSubgroupIndex ) );

		if ( isValidSubgroupIndex ) {

			WorkgroupArray_898[ subgroupIndex ] = t;


		}

		workgroupBarrier();
		workgroupSectionOffset = ( workgroupSectionOffset + subgroupSizeLog );

	}


	if ( ( invocationLocalIndex == 0u ) ) {

		Prefix_Sum_Reduction_0.value[ workgroupId.x ] = WorkgroupArray_898[ ( spineSize - 1u ) ];


	}

}
`;
};
