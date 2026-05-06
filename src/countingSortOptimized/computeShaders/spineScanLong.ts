import { constructBuiltinDeclarations } from './utils';

export const SpineScanLongCompute = (
  workgroupSize: number,
  linearIndexingAvailable: boolean = false,
  subgroupIDAvailable: boolean = false
): string => {
  const builtinDeclarations = constructBuiltinDeclarations(
    linearIndexingAvailable,
	subgroupIDAvailable
  );

  return /* wgsl */ `
${!linearIndexingAvailable ? 'var<private> instanceIndex : u32;' : ''}

var<workgroup> WorkgroupArray_898: array< u32, ${workgroupSize / 4} >;

@group( 0 ) @binding( 1 )
var<storage, read_write> Prefix_Sum_Reduction_0 : U32ArrayStruct;

@group(1) @binding(0) var<uniform> params: PrefixSumParams;

@compute @workgroup_size( ${workgroupSize}, 1, 1 )
fn spineScanLong(
${builtinDeclarations}
) {

	// system
	${!linearIndexingAvailable ? `instanceIndex = globalId.x + globalId.y * ( ${workgroupSize} * numWorkgroups.x ) + globalId.z * ( ${workgroupSize} * numWorkgroups.x ) * ( 1 * numWorkgroups.y );` : ''}

	// vars

	var subgroupSizeLog : u32;
	var spineSize : u32;
	var spineSizeLog : u32;
	var nodeVar0 : u32;
	var subgroupAlignedSize : u32;
	var spinePartitionSize : u32;
	var spineAlignedSize : u32;
	var nodeVar1 : array< u32, 16 >;
	var previousReduction : u32;
	var unvectorizedSubgroupOffset : u32;
	var s_offset : u32;
	var nodeVar2 : u32;
	var prev : u32;
	var nodeVar3 : u32;
	var nodeVar4 : u32;
	var nodeVar7 : bool;
	var nodeVar8 : u32;
	var nodeVar9 : bool;
	var nodeVar10 : u32;
	var nodeVar11 : u32;
	var nodeVar12 : u32;
	var nodeVar13 : bool;
	var nodeVar15 : u32;

	subgroupSizeLog = countTrailingZeros( subgroupSize );
	spineSize = ( params.workgroupSize >> subgroupSizeLog );
	spineSizeLog = countTrailingZeros( spineSize );
	nodeVar0 = ( ( spineSizeLog + subgroupSizeLog ) - 1u );
	nodeVar0 = ( nodeVar0 / subgroupSizeLog );
	nodeVar0 = ( nodeVar0 * subgroupSizeLog );
	subgroupAlignedSize = ( 1u << nodeVar0 );
    var unvectorizedWorkPerInvocation = params.workPerInvocation * 4u;
	spinePartitionSize = params.workgroupSize * unvectorizedWorkPerInvocation;
	spineAlignedSize = ( ( spinePartitionSize + params.workgroupCount ) - 1u );
	spineAlignedSize = ( spineAlignedSize / spinePartitionSize );
	spineAlignedSize = ( spineAlignedSize * spinePartitionSize );
	nodeVar1 = array< u32, 16 >( 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u );
	previousReduction = 0u;
	${!subgroupIDAvailable ? 'var invocationSubgroupMetaIndex: u32 = ( invocationLocalIndex / subgroupSize );' : ''}
	unvectorizedSubgroupOffset = ( ( invocationSubgroupMetaIndex * subgroupSize ) * 16u );
	s_offset = ( unvectorizedSubgroupOffset + invocationSubgroupIndex );

	for ( var j : u32 = 0u; j < spineAlignedSize; j += spinePartitionSize ) {

		nodeVar2 = ( s_offset + j );

		for ( var k : u32 = 0u; k < unvectorizedWorkPerInvocation; k ++ ) {


			if ( nodeVar2 < params.workgroupCount ) {

				nodeVar1[ k ] = Prefix_Sum_Reduction_0.value[ nodeVar2 ];


			}

			nodeVar2 = ( nodeVar2 + subgroupSize );

		}

		prev = 0u;

		for ( var k : u32 = 0u; k < unvectorizedWorkPerInvocation; k += 1u ) {

			nodeVar1[ k ] = ( subgroupInclusiveAdd( nodeVar1[ k ] ) + prev );
			prev = subgroupShuffle( nodeVar1[ k ], ( subgroupSize - 1u ) );

		}

		WorkgroupArray_898[ invocationSubgroupMetaIndex ] = prev;
		workgroupBarrier();
		nodeVar3 = 0u;
		nodeVar4 = 0u;

		for ( var j : u32 = subgroupSize; j <= subgroupAlignedSize; j <<= subgroupSizeLog ) {

			nodeVar7 = ( j != subgroupSize );
			nodeVar8 = ( ( ( invocationLocalIndex + nodeVar3 ) << nodeVar4 ) - select( 0u, 1u, nodeVar7 ) );
			nodeVar9 = ( nodeVar8 < spineSize );
			nodeVar10 = subgroupInclusiveAdd( select( 0u, WorkgroupArray_898[ nodeVar8 ], nodeVar9 ) );

			if ( nodeVar9 ) {

				WorkgroupArray_898[ nodeVar8 ] = nodeVar10;

			}

			if ( nodeVar7 ) {

				nodeVar11 = ( j >> subgroupSizeLog );
				nodeVar12 = ( invocationLocalIndex + nodeVar11 );

				if ( ( ( nodeVar12 & ( j - 1u ) ) >= nodeVar11 ) ) {

					nodeVar13 = ( nodeVar12 < spineSize );

					if ( ( nodeVar13 && ( ( ( nodeVar12 + 1u ) & ( nodeVar11 - 1u ) ) != 0u ) ) ) {

						WorkgroupArray_898[ nodeVar12 ] = ( WorkgroupArray_898[ nodeVar12 ] + select( 0u, WorkgroupArray_898[ ( ( ( nodeVar12 >> nodeVar4 ) << nodeVar4 ) - 1u ) ], nodeVar13 ) );


					}

				}

			} else {

				nodeVar3 = ( nodeVar3 + 1u );

			}

			nodeVar4 = ( nodeVar4 + subgroupSizeLog );

		}

		workgroupBarrier();
		nodeVar15 = ( s_offset + j );

		for ( var k : u32 = 0u; k < unvectorizedWorkPerInvocation; k ++ ) {


			if ( nodeVar15 < params.workgroupCount ) {

				Prefix_Sum_Reduction_0.value[ nodeVar15 ] = ( nodeVar1[ k ] + ( select( 0u, WorkgroupArray_898[ ( invocationSubgroupMetaIndex - 1u ) ], ( invocationSubgroupMetaIndex != 0u ) ) + previousReduction ) );

			}

			nodeVar15 = ( nodeVar15 + subgroupSize );

		}

		previousReduction = ( previousReduction + subgroupBroadcast( WorkgroupArray_898[ ( subgroupAlignedSize - 1u ) ], 0u ) );
		workgroupBarrier();

	}

}
`;
};
