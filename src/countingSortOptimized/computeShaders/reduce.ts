import { constructBuiltinDeclarations } from './utils';

export const ReduceCompute = (
  workgroupSize: number,
  linearIndexingAvailable: boolean
): string => {
  const builtinDeclarations = constructBuiltinDeclarations(
    linearIndexingAvailable
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

	${
    !linearIndexingAvailable &&
    `instanceIndex = globalId.x + globalId.y * ( ${workgroupSize} * numWorkgroups.x ) + globalId.z * ( ${workgroupSize} * numWorkgroups.x ) * ( 1 * numWorkgroups.y );`
  }

    // TODO: Investigate replacing with subgroup_id
	var invocationSubgroupMetaIndex : u32;
	var subgroupOffset : u32;
	var threadSubgroupOffset : u32;
	var workgroupOffset : u32;
	var startThreadBase : u32;
	var startThread : u32;
	var nodeVar0 : u32;
	var nodeVar1 : vec4<u32>;
	var subgroupSizeLog : u32;
	var spineSize : u32;
	var spineSizeLog : u32;
	var nodeVar2 : u32;
	var subgroupAlignedSize : u32;
	var nodeVar3 : u32;
	var nodeVar4 : u32;
	var isValidSubgroupIndex : bool;
	var nodeVar5 : u32;
	var t : u32;

	invocationSubgroupMetaIndex = ( invocationLocalIndex / subgroupSize );
	// subgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * 4u );
    subgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * params.workPerInvocation );
	threadSubgroupOffset = ( subgroupOffset + invocationSubgroupIndex );
	// workgroupOffset = ( workgroupId.x * ( 256u * 4u ) );
    workgroupOffset = ( workgroupId.x * ( params.workgroupSize * params.workPerInvocation ) );
	startThreadBase = ( threadSubgroupOffset + workgroupOffset );
	startThread = startThreadBase;
	nodeVar0 = 0u;

	// if ( ( workgroupId.x < ( 8u - 1u ) ) ) {
    if ( ( workgroupId.x < ( params.workgroupCount - 1u ) ) ) {

		// for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < 4u; currentSubgroupInBlock ++ ) {
        for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			nodeVar0 = ( nodeVar0 + u32( dot( vec4<u32>( 1u, 1u, 1u, 1u ), Prefix_Sum_Input_Vec_0.value[ startThread ] ) ) );
			startThread = ( startThread + subgroupSize );

		}

	}


	// if ( ( workgroupId.x == ( 8u - 1u ) ) ) {
    if ( ( workgroupId.x == ( params.workgroupCount - 1u ) ) ) {


		for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			nodeVar0 = ( nodeVar0 + u32( dot( select( vec4<u32>( 0u, 0u, 0u, 0u ), Prefix_Sum_Input_Vec_0.value[ startThread ], ( startThread < params.vecCount ) ), vec4<u32>( 1u, 1u, 1u, 1u ) ) ) );
			startThread = ( startThread + subgroupSize );

		}



	}

	nodeVar0 = subgroupAdd( nodeVar0 );

	if ( ( invocationSubgroupIndex == 0u ) ) {

		WorkgroupArray_898[ invocationSubgroupMetaIndex ] = nodeVar0;


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
	nodeVar3 = 0u;

	for ( var j : u32 = subgroupSize; j <= subgroupAlignedSize; j <<= subgroupSizeLog ) {

		nodeVar4 = ( ( ( invocationLocalIndex + 1u ) << nodeVar3 ) - 1u );
		isValidSubgroupIndex = ( nodeVar4 < spineSize );
		t = subgroupAdd( select( 0u, WorkgroupArray_898[ nodeVar4 ], isValidSubgroupIndex ) );

		if ( isValidSubgroupIndex ) {

			WorkgroupArray_898[ nodeVar4 ] = t;


		}

		workgroupBarrier();
		nodeVar3 = ( nodeVar3 + subgroupSizeLog );

	}


	if ( ( invocationLocalIndex == 0u ) ) {

		Prefix_Sum_Reduction_0.value[ workgroupId.x ] = WorkgroupArray_898[ ( spineSize - 1u ) ];


	}

}
`;
};
