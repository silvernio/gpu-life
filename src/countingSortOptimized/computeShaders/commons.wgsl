// directives
enable subgroups;

struct Vec4ArrayStruct {
	value: array<vec4<u32>>
}

struct U32ArrayStruct {
	value: array<u32>
}

struct PrefixSumParams {
    workgroupCount: u32,
    workgroupSize: u32,
    vecCount: u32,
    workPerInvocation: u32,
}