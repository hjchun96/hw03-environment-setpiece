#version 300 es
precision highp float;

// The vertex shader used to render the background of the scene
uniform float u_Time;

in vec4 vs_Pos;
out vec2 fs_Pos;
out vec4 fs_LightVec;

const vec4 lightPos = vec4(8, 2, 9, 1);

void main() {

  fs_Pos = vs_Pos.xy;
  fs_LightVec = lightPos;  // Compute the direction in which the light source lies
  gl_Position = vs_Pos;
}
