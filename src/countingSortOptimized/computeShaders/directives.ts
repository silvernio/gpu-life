export const directivesCode = (
    hasLinearIndexing: boolean,
    hasSubgroupID: boolean
) => {

return /*wgsl*/`
// directives
${hasLinearIndexing ? 'requires linear_indexing;' : ''}
${hasSubgroupID ? 'requires subgroup_id;' : ''}
enable subgroups;
`;
}