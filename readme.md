
![banner](<screenshots/banner.png>)

<h1 align="center">GPU Life</h1>
A little simulation of particle life made using webgpu and typescript.

The algortihm for the particle movement is entirely based of particle life. It's just been moved onto a compute shader to allow for larger numbers of particles.

The movement of the particles is done using 2 ping pong buffers going through a compute shader. Then the rendering is done using a basic vertex and fragment shader each frame.

There are 3 engines that change how the particles are managed:
- **NSquared** - each particle checks every other particle, this is highly inefficient.
- **Atomic Linked Lists** - particles are put into cells, where they are added to lists, then each particle can only check other particles in nearby cells to improve performance in larger environments.
- **Counting Sort** - once particles are put into cells, the array that stores their cell ids are sorted directly. This approach results in faster reading of nearby particles, at the cost of a slower construction time.
- **Prefix Sum** - A modified version of the counting sorting algorithm written by <a href="https://www.github.com/cmhhelgeson">cmhhelgeson</a> that replaces the serial exclusive prefix sum algorithm with a parallel inclusive prefix sum adapted from <a href="https://github.com/b0nes164/GPUPrefixSums">B0nes164's Prefix Sum Algorithms</a>. In simulations with tens of thousands of cells, the prefix sum algorithm will no longer act as a bottleneck for performance, with execution times reduced from ~100ms to ~0.3ms.
