
//---------- SDF Functions ----------//
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf
}


//---------- Raymarching Core ----------//

float raymarch(vec3 eye, vec3 raymarchDir, float start, float end) {// eye = ray orientation

	/*//BVH Optimziation
	Cube objectCube = createBikeCube();
	Cube floorCube = createBlockCube();
	bool hitCube = rayCubeIntersect(eye, raymarchDir, objectCube);
	bool hitPlane = rayCubeIntersect(eye, raymarchDir, floorCube);
	if (!hitPlane && ! hitCube) return end;*/

	float depth = start;
	for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
		vec3 pnt = eye + depth * raymarchDir;
		float dist = sceneSDF(pnt);

		if (dist < EPSILON) {
			if (dist  == plane_dist) {
				IS_PLANE = 1;
			}
			return depth;
		}
		depth += dist; // Move along the view ray, spheremarch optimization
		if (depth >= end) { // abort
			return end;
		}
	}
	return end;
}

vec3 calculateRayMarchPoint() {
  vec3 forward = u_Ref - u_Eye;
	vec3 right = normalize(cross(u_Up, forward));
	vec3 up = normalize(cross(forward, right));
	float len = length(u_Ref - u_Eye);
	forward = normalize(forward);
	float aspect = u_Dimensions.x / u_Dimensions.y;

  float fovy = 90.0;
	float alpha = radians(fovy/2.0);
	vec3 V = up * len * tan(alpha);
	vec3 H = right * len * aspect * tan(alpha);

	float sx = 1.0 - (2.0 * gl_FragCoord.x/u_Dimensions.x);
	float sy = (2.0 * gl_FragCoord.y/u_Dimensions.y) - 1.0;
	vec3 p = u_Ref + sx * H + sy * V;
	return p;
}


//---------- Main ----------//
    mat3 setCamera( in vec3 ro, in vec3 rt, in float cr )
{
    vec3 cw = normalize(rt-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, -cw );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )

{
      /*
    vec2  p = (-iResolution.xy+2.0*fragCoord.xy)/iResolution.y;
    vec2  q = fragCoord.xy/iResolution.xy;
    float an = 1.87 - 0.04*(1.0-cos(0.5*iTime));

  vec3  ro = vec3(-0.4,0.2,0.0) + 2.2*vec3(cos(an),0.0,sin(an));
   	vec3  ta = vec3(-0.6,0.2,0.0);
    mat3  ca = setCamera( ro, ta, 0.0 );
    vec3  rd = normalize( ca * vec3(p,-2.8) );
    vec3 col = render( ro, rd, q );
    fragColor = vec4( col, 1.0 ); */


    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    //vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    vec3 col = vec3(255.0/255.0, 218.0/255.0, 26.0/255.0);
    // Output to screen
    fragColor = vec4(col,1.0);




}
