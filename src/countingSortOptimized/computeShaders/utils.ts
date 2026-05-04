export const constructBuiltinDeclarations = (
  linearIndexingAvailable: boolean = false,
  subgroupIDAvailable: boolean = false
) => {
  const builtinManifest = [
    {
      builtin: 'local_invocation_index',
      label: 'invocationLocalIndex',
      type: 'u32',
    },
    {
      builtin: 'subgroup_invocation_id',
      label: 'invocationSubgroupIndex',
      type: 'u32',
    },

    {
      builtin: 'global_invocation_id',
      label: 'globalId',
      type: 'vec3<u32>',
    },
    {
      builtin: 'workgroup_id',
      label: 'workgroupId',
      type: 'vec3<u32>',
    },
    {
      builtin: 'local_invocation_id',
      label: 'localId',
      type: 'vec3<u32>',
    },
    {
      builtin: 'subgroup_size',
      label: 'subgroupSize',
      type: 'u32',
    },
  ];

  if (linearIndexingAvailable) {
    builtinManifest.push({
      builtin: 'global_invocation_index',
      label: 'instanceIndex',
      type: 'u32',
    });
  } else {
    builtinManifest.push({
      builtin: 'num_workgroups',
      label: 'numWorkgroups',
      type: 'vec3<u32>',
    });
  }

  if (subgroupIDAvailable) {
    builtinManifest.push({
      builtin: 'subgroup_id',
      label: 'invocationSubgroupMetaIndex',
      type: 'u32'
    })
  }

  const builtinDeclarations = builtinManifest
    .map((manifest) => {
      return `\t@builtin( ${manifest.builtin} ) ${manifest.label} : ${manifest.type},`;
    })
    .join('\n');

  return builtinDeclarations;
};
