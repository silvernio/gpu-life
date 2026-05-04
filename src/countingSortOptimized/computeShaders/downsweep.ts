import { constructBuiltinDeclarations } from './utils';

export const DownSweepCompute = (
  workgroupSize: number,
  linearIndexingAvailable: boolean = false,
  subgroupIDAvailable: boolean = false,
): string => {
  const builtinDeclarations = constructBuiltinDeclarations(
    linearIndexingAvailable,
	subgroupIDAvailable
  );
  return /* wgsl */ `
override OUTPUT_INDEX_OFFSET: u32 = 0u;

${!linearIndexingAvailable ? 'var<private> instanceIndex : u32;' : ''}

var<workgroup> WorkgroupArray_898: array< u32, ${workgroupSize / 4} >;

@group( 0 ) @binding(0)
var<storage, read_write> Prefix_Sum_Input_Vec_0 : Vec4ArrayStruct;

@group( 0 ) @binding(1)
var<storage, read_write> Prefix_Sum_Reduction_0 : U32ArrayStruct;

// Original vec4 output binding (standard prefix sum, no offset):
// @group(0) @binding( 2 )
// var<storage, read_write> Prefix_Sum_Output_Vec_0 : Vec4ArrayStruct;
@group( 0 ) @binding(2)
var<storage, read_write> prefix_sum_output : U32ArrayStruct;


// Uniforms in seperate bind group since they get changed less often

@group(1) @binding(0) var<uniform> params: PrefixSumParams;

@compute @workgroup_size( ${workgroupSize}, 1, 1 )
fn downSweep(
${builtinDeclarations}
) {

	${!linearIndexingAvailable ? `instanceIndex = globalId.x + globalId.y * ( ${workgroupSize} * numWorkgroups.x ) + globalId.z * ( ${workgroupSize} * numWorkgroups.x ) * ( 1 * numWorkgroups.y );` : ''}

	var subgroupOffset : u32;
	var workgroupOffset : u32;
	var nodeVar0 : u32;
	var nodeVar1 : u32;
	var nodeVar2 : array< vec4<u32>, 4 >;
	var nodeVar3 : u32;
	var laneMask : u32;
	var clockwiseShift : u32;
	var prevAccGreatestValue : u32;
	var nodeVar5 : u32;
	var nodeVar6 : u32;
	var subgroupSizeLog : u32;
	var spineSize : u32;
	var spineSizeLog : u32;
	var nodeVar7 : u32;
	var subgroupAlignedSize : u32;
	var nodeVar9 : u32;
	var nodeVar10 : bool;
	var nodeVar11 : u32;
	var nodeVar12 : u32;
	var nodeVar13 : u32;
	var nodeVar14 : bool;
	var savedX : array< u32, 4 >;

	${!subgroupIDAvailable ? 'var invocationSubgroupMetaIndex: u32 = ( invocationLocalIndex / subgroupSize );' : ''}
	subgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * params.workPerInvocation );
	workgroupOffset = ( workgroupId.x * ( params.workgroupSize * params.workPerInvocation ) );
	nodeVar0 = ( ( subgroupOffset + invocationSubgroupIndex ) + workgroupOffset );
	nodeVar1 = nodeVar0;
	nodeVar2 = array< vec4<u32>, 4 >( vec4<u32>( 0u, 0u, 0u, 0u ), vec4<u32>( 0u, 0u, 0u, 0u ), vec4<u32>( 0u, 0u, 0u, 0u ), vec4<u32>( 0u, 0u, 0u, 0u ) );

	if ( ( workgroupId.x < ( params.workgroupCount - 1u ) ) ) {

		for (
            var currentSubgroupInBlock : u32 = 0u;
            currentSubgroupInBlock < params.workPerInvocation;
            currentSubgroupInBlock ++
        ) {

			nodeVar2[ currentSubgroupInBlock ] = Prefix_Sum_Input_Vec_0.value[ nodeVar1 ];
			nodeVar2[ currentSubgroupInBlock ].y = ( nodeVar2[ currentSubgroupInBlock ].y + nodeVar2[ currentSubgroupInBlock ].x );
			nodeVar2[ currentSubgroupInBlock ].z = ( nodeVar2[ currentSubgroupInBlock ].z + nodeVar2[ currentSubgroupInBlock ].y );
			nodeVar2[ currentSubgroupInBlock ].w = ( nodeVar2[ currentSubgroupInBlock ].w + nodeVar2[ currentSubgroupInBlock ].z );
			nodeVar1 = ( nodeVar1 + subgroupSize );

		}

	}

	if ( ( workgroupId.x == ( params.workgroupCount - 1u ) ) ) {

		for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			if ( ( nodeVar1 < params.vecCount ) ) {

				nodeVar2[ currentSubgroupInBlock ] = Prefix_Sum_Input_Vec_0.value[ nodeVar1 ];
				nodeVar2[ currentSubgroupInBlock ].y = ( nodeVar2[ currentSubgroupInBlock ].y + nodeVar2[ currentSubgroupInBlock ].x );
				nodeVar2[ currentSubgroupInBlock ].z = ( nodeVar2[ currentSubgroupInBlock ].z + nodeVar2[ currentSubgroupInBlock ].y );
				nodeVar2[ currentSubgroupInBlock ].w = ( nodeVar2[ currentSubgroupInBlock ].w + nodeVar2[ currentSubgroupInBlock ].z );
				nodeVar1 = ( nodeVar1 + subgroupSize );

			}

		}

	}

	nodeVar3 = 0u;
	laneMask = ( subgroupSize - 1u );
	clockwiseShift = ( ( invocationSubgroupIndex + laneMask ) & laneMask );

	for (
        var currentSubgroupInBlock : u32 = 0u;
        currentSubgroupInBlock < params.workPerInvocation;
        currentSubgroupInBlock ++
    ) {

		prevAccGreatestValue = subgroupShuffle( subgroupInclusiveAdd( nodeVar2[ currentSubgroupInBlock ].w ), clockwiseShift );
		nodeVar2[ currentSubgroupInBlock ] = ( nodeVar2[ currentSubgroupInBlock ] + ( vec4<u32>( nodeVar3 ) + select( vec4<u32>( 0u, 0u, 0u, 0u ), vec4<u32>( prevAccGreatestValue ), ( invocationSubgroupIndex != 0u ) ) ) );
		nodeVar3 = ( nodeVar3 + subgroupBroadcast( prevAccGreatestValue, 0u ) );

	}


	if ( ( invocationSubgroupIndex == 0u ) ) {

		WorkgroupArray_898[ invocationSubgroupMetaIndex ] = nodeVar3;

	}

	workgroupBarrier();
	nodeVar5 = 0u;
	nodeVar6 = 0u;
	subgroupSizeLog = countTrailingZeros( subgroupSize );
	spineSize = ( params.workgroupSize >> subgroupSizeLog );
	spineSizeLog = countTrailingZeros( spineSize );
	nodeVar7 = ( ( spineSizeLog + subgroupSizeLog ) - 1u );
	nodeVar7 = ( nodeVar7 / subgroupSizeLog );
	nodeVar7 = ( nodeVar7 * subgroupSizeLog );
	subgroupAlignedSize = ( 1u << nodeVar7 );

	for ( var j : u32 = subgroupSize; j <= subgroupAlignedSize; j <<= subgroupSizeLog ) {

		nodeVar9 = ( ( ( invocationLocalIndex + nodeVar5 ) << nodeVar6 ) - nodeVar5 );
		nodeVar10 = ( nodeVar9 < spineSize );
		nodeVar11 = subgroupInclusiveAdd( select( 0u, WorkgroupArray_898[ nodeVar9 ], nodeVar10 ) );

		if ( nodeVar10 ) {

			WorkgroupArray_898[ nodeVar9 ] = nodeVar11;


		}

		workgroupBarrier();

		if ( ( j != subgroupSize ) ) {

			nodeVar12 = ( j >> subgroupSizeLog );
			nodeVar13 = ( invocationLocalIndex + nodeVar12 );

			if ( ( ( nodeVar13 & ( j - 1u ) ) >= nodeVar12 ) ) {

				nodeVar14 = ( nodeVar13 < spineSize );

				if ( ( nodeVar14 && ( ( ( nodeVar13 + 1u ) & ( nodeVar12 - 1u ) ) != 0u ) ) ) {

					WorkgroupArray_898[ nodeVar13 ] = ( WorkgroupArray_898[ nodeVar13 ] + select( 0u, WorkgroupArray_898[ ( ( ( nodeVar13 >> nodeVar6 ) << nodeVar6 ) - 1u ) ], nodeVar14 ) );

				}

			}

		} else {

			nodeVar5 = ( nodeVar5 + 1u );

		}

		nodeVar6 = ( nodeVar6 + subgroupSize );

	}

	workgroupBarrier();
	nodeVar3 = ( select( 0u, Prefix_Sum_Reduction_0.value[ ( workgroupId.x - 1u ) ], ( workgroupId.x != 0u ) ) + select( 0u, WorkgroupArray_898[ ( invocationSubgroupMetaIndex - 1u ) ], ( invocationSubgroupMetaIndex != 0u ) ) );
	nodeVar1 = nodeVar0;

	if ( ( workgroupId.x < ( params.workgroupCount - 1u ) ) ) {

		for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			// Original vec4 write (standard prefix sum, no offset):
			// Prefix_Sum_Output_Vec_0.value[ nodeVar1 ] = ( nodeVar2[ currentSubgroupInBlock ] + vec4<u32>( nodeVar3 ) );

			// nodeVar1 is the vec4 index; multiply by 4 to get the u32 index of the .x component.
			let indexU32Base = nodeVar1 * 4u;
			let outputValueToWrite = ( nodeVar2[ currentSubgroupInBlock ] + vec4<u32>( nodeVar3 ) );
			prefix_sum_output.value[ indexU32Base + 0u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.x;
			prefix_sum_output.value[ indexU32Base + 1u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.y;
			prefix_sum_output.value[ indexU32Base + 2u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.z;
			prefix_sum_output.value[ indexU32Base + 3u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.w;
			nodeVar1 = ( nodeVar1 + subgroupSize );
		}

	}

	if ( ( workgroupId.x == ( params.workgroupCount - 1u ) ) ) {

		for ( var currentSubgroupInBlock : u32 = 0u; currentSubgroupInBlock < params.workPerInvocation; currentSubgroupInBlock ++ ) {

			if ( ( nodeVar1 < params.vecCount ) ) {

				// Original vec4 write (standard prefix sum, no offset):
				// Prefix_Sum_Output_Vec_0.value[ nodeVar1 ] = ( nodeVar2[ currentSubgroupInBlock ] + vec4<u32>( nodeVar3 ) );

				let indexU32Base = nodeVar1 * 4u;
				let outputValueToWrite = ( nodeVar2[ currentSubgroupInBlock ] + vec4<u32>( nodeVar3 ) );
				prefix_sum_output.value[ indexU32Base + 0u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.x;
				prefix_sum_output.value[ indexU32Base + 1u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.y;
				prefix_sum_output.value[ indexU32Base + 2u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.z;
				prefix_sum_output.value[ indexU32Base + 3u + OUTPUT_INDEX_OFFSET ] = outputValueToWrite.w;
				nodeVar1 = ( nodeVar1 + subgroupSize );

			}
		}
	}

	workgroupBarrier();

	// Zero the element at [OUTPUT_INDEX_OFFSET - 1] so sort.wgsl can use it as the
	// exclusive-prefix-sum starting offset for cell 0 on the next tick (sort will have
	// incremented it to counts[0] during the previous tick).
	if ( OUTPUT_INDEX_OFFSET > 0u && workgroupId.x == 0u && invocationLocalIndex == 0u ) {
		for ( var x : u32 = 0u; x < OUTPUT_INDEX_OFFSET; x ++ ) {
			prefix_sum_output.value[ x ] = 0u;
		}
	}
}
`;
};
