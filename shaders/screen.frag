#version 460

out vec4 outColor;

layout (location = 0) in vec2 inTexCoords;

#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define SAMPLES 8
#define MAX_BOUNCES 8

uniform int seed;
uniform int debugView;
uniform bool useSkybox;
uniform sampler2D skybox;

bool useCachedColor = false;
vec3 cachedColor = vec3(0.0);

struct Plane {
    vec3 pos;
    float pad1;
    uint materialIndex;
    uint axis;
    vec2 size;
    vec2 padding;
    uint padding2[2];
};

struct Vertex {
    vec3 position;
    float padding;
};

struct Triangle {
    vec3 color;
    uint materialIndex;
    Vertex v0;
    Vertex v1;
    Vertex v2;
};

struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 energy;
};

struct HitReport {
    float distance;
    vec3 position;
    vec3 normal;
    bool fromInside;
    vec2 texCoords;
    uint materialIndex;
};

struct Sphere {
    vec3 position;
    float radius;
    uint materialIndex;
    float padding1;
};

struct Material {
    vec3 albedo;
    float padding1;
    vec3 specularColor;
    float padding2;
    vec3 emissive;
    float padding3;
    float directSpecular;
    float indirectSpecular;
    float specularRoughness;
    float refractiveIndex;
    float refractionChance;
    float refractiveRoughness;
    vec3 refractiveColor;
    float padding4;
};

struct BVHNode {
    vec3 min;
    uint leftChildOrFirstTriangle;
    vec3 max;
    uint triangleCount;
};

layout (std140, binding = 0) uniform Camera {
    mat4 inverseView;
    mat4 inverseProjection;
    vec2 resolution;
    vec3 position;
} camera;

layout (std430, binding = 1) buffer Scene {
    uint sphereCount;
    uint pad1[3];
    Sphere spheres[100];

    uint planeCount;
    uint pad4[3];
    Plane planes[100];

    uint triangleCount;
    uint pad5[3];
    Triangle triangles[100];

    uint meshCount;
    uint pad6[3];

    uint materialCount;
    uint pad7[3];
    Material materials[100];
} scene;

layout (std430, binding = 2) buffer Mesh1 {
    uint triangleCount;
    uint pad1[3];
    mat4 transform;
    Triangle triangles[];
} mesh1;

layout  (std430, binding = 3) buffer Mesh1BVH {
    BVHNode nodes[];
} mesh1BVH;

layout (std430, binding = 4) buffer Mesh1PrimIndices {
    uint primIndices[];
} mesh1PrimIndices;

layout (std430, binding = 5) buffer Mesh2 {
    uint triangleCount;
    uint pad1[3];
    mat4 transform;
    Triangle triangles[];
} mesh2;

layout  (std430, binding = 6) buffer Mesh2BVH {
    BVHNode nodes[];
} mesh2BVH;

layout (std430, binding = 7) buffer Mesh2PrimIndices {
    uint primIndices[];
} mesh2PrimIndices;

layout (std430, binding = 8) buffer Mesh3 {
    uint triangleCount;
    uint pad1[3];
    mat4 transform;
    Triangle triangles[];
} mesh3;

layout  (std430, binding = 9) buffer Mesh3BVH {
    BVHNode nodes[];
} mesh3BVH;

layout (std430, binding = 10) buffer Mesh3PrimIndices {
    uint primIndices[];
} mesh3PrimIndices;

layout (std430, binding = 11) buffer Mesh4 {
    uint triangleCount;
    uint pad1[3];
    mat4 transform;
    Triangle triangles[];
} mesh4;

layout  (std430, binding = 12) buffer Mesh4BVH {
    BVHNode nodes[];
} mesh4BVH;

layout (std430, binding = 13) buffer Mesh4PrimIndices {
    uint primIndices[];
} mesh4PrimIndices;

layout (std430, binding = 14) buffer Mesh5 {
    uint triangleCount;
    uint pad1[3];
    mat4 transform;
    Triangle triangles[];
} mesh5;

layout  (std430, binding = 15) buffer Mesh5BVH {
    BVHNode nodes[];
} mesh5BVH;

layout (std430, binding = 16) buffer Mesh5PrimIndices {
    uint primIndices[];
} mesh5PrimIndices;

#define PLANE_AXIS_X 0
#define PLANE_AXIS_Y 1
#define PLANE_AXIS_Z 2

bool testSphereIntersection(Ray ray, inout HitReport hitReport, uint sphereIndex) {
	vec3 originToCenter = ray.origin - scene.spheres[sphereIndex].position;

	float b = dot(originToCenter, ray.direction);
	float c = dot(originToCenter, originToCenter) - scene.spheres[sphereIndex].radius * scene.spheres[sphereIndex].radius;
	if (c > 0.0 && b > 0.0)
		return false;

	float discr = b * b - c;
	if (discr < 0.0)
		return false;

    bool fromInside = false;
    float hitDistance = -b - sqrt(discr);
    if (hitDistance < 0.0) {
        fromInside = true;
        hitDistance = -b + sqrt(discr);
    }

    if ((hitDistance > 0.01 || hitDistance < 0.01) && hitDistance < hitReport.distance) {
        hitReport.distance = hitDistance;
        hitReport.position = ray.origin + ray.direction * hitDistance;
        hitReport.fromInside = fromInside;
        hitReport.normal = normalize(hitReport.position - scene.spheres[sphereIndex].position) * (fromInside ? -1.0 : 1.0);
        hitReport.materialIndex = scene.spheres[sphereIndex].materialIndex;
    }

    return true;
}

bool testPlaneIntersection(Ray ray, inout HitReport hitReport, uint planeIndex) {
    Plane plane = scene.planes[planeIndex];
    vec3 normal = plane.axis == PLANE_AXIS_X ? vec3(1, 0, 0) : plane.axis == PLANE_AXIS_Y ? vec3(0, 1, 0) : vec3(0, 0, 1);

    float denom = dot(ray.direction, normal);
    if (abs(denom) > 0.0001) {
        float distance = dot(plane.pos - ray.origin, normal) / denom;

        if (distance > 0.0001 && distance < hitReport.distance) {
            vec3 hitPos = ray.origin + ray.direction * distance;

            bool hit = false;
            if (plane.axis == PLANE_AXIS_Y && hitPos.x > plane.pos.x - plane.size.x / 2 && hitPos.x < plane.pos.x + plane.size.x / 2 && hitPos.z > plane.pos.z - plane.size.y / 2 && hitPos.z < plane.pos.z + plane.size.y / 2) {
                hit = true;
                if (ray.origin.y < plane.pos.y)
                    normal.y = -normal.y;
            } else if (plane.axis == PLANE_AXIS_X && hitPos.y > plane.pos.y - plane.size.x / 2 && hitPos.y < plane.pos.y + plane.size.x / 2 && hitPos.z > plane.pos.z - plane.size.y / 2 && hitPos.z < plane.pos.z + plane.size.y / 2) {
                hit = true;
                if (ray.origin.x < plane.pos.x)
                    normal.x = -normal.x;
            } else if (plane.axis == PLANE_AXIS_Z && hitPos.x > plane.pos.x - plane.size.x / 2 && hitPos.x < plane.pos.x + plane.size.x / 2 && hitPos.y > plane.pos.y - plane.size.y / 2 && hitPos.y < plane.pos.y + plane.size.y / 2) {
                hit = true;
                if (ray.origin.z < plane.pos.z)
                    normal.z = -normal.z;
            }

            if (hit) {
                hitReport.distance = distance;
                hitReport.position = hitPos;
                hitReport.normal = normal;
                hitReport.fromInside = false;
                hitReport.materialIndex = plane.materialIndex;

                {   // Calculate custom UV
                    vec3 localPos = hitPos - plane.pos;
                    if (plane.axis == PLANE_AXIS_X) {
                        localPos.y += plane.size.x / 2;
                        localPos.z += plane.size.y / 2;
                        hitReport.texCoords = vec2(localPos.z / plane.size.y, localPos.y / plane.size.x);
                    } else if (plane.axis == PLANE_AXIS_Y) {
                        localPos.x += plane.size.x / 2;
                        localPos.z += plane.size.y / 2;
                        hitReport.texCoords = vec2(localPos.x / plane.size.x, localPos.z / plane.size.y);
                    } else {
                        localPos.x += plane.size.x / 2;
                        localPos.y += plane.size.y / 2;
                        hitReport.texCoords = vec2(localPos.x / plane.size.x, localPos.y / plane.size.y);
                    }
                }

                return true;
            }
        }
    }

    return false;
}

bool testTriangleIntersection(Ray ray, inout HitReport hitReport, uint triangleIndex, uint meshIndex, bool isMesh) {
    Triangle triangle;
    mat4 transform = mat4(1.0);

    if (isMesh) {
        if (meshIndex == 0) {
            triangle = mesh1.triangles[triangleIndex];
            transform = mesh1.transform;
        } else if (meshIndex == 1) {
            triangle = mesh2.triangles[triangleIndex];
            transform = mesh2.transform;
        } else if (meshIndex == 2) {
            triangle = mesh3.triangles[triangleIndex];
            transform = mesh3.transform;
        } else if (meshIndex == 3) {
            triangle = mesh4.triangles[triangleIndex];
            transform = mesh4.transform;
        } else if (meshIndex == 4) {
            triangle = mesh5.triangles[triangleIndex];
            transform = mesh5.transform;
        }
    } else
        triangle = scene.triangles[triangleIndex];

    vec3 v0 = (transform * vec4(triangle.v0.position, 1.0)).xyz;
    vec3 v1 = (transform * vec4(triangle.v1.position, 1.0)).xyz;
    vec3 v2 = (transform * vec4(triangle.v2.position, 1.0)).xyz;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(ray.direction, edge2);

    float a = dot(edge1, h);
    if (a > -0.0001 && a < 0.0001)
        return false;

    float f = 1.0 / a;
    vec3 s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    vec3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * dot(edge2, q);
    if (t > 0.0001 && t < hitReport.distance) {
        hitReport.distance = t;
        hitReport.position = ray.origin + ray.direction * t;
        mat3 normalMatrix = transpose(inverse(mat3(transform)));
        hitReport.normal = -normalize(normalMatrix * cross(edge1, edge2));
        hitReport.fromInside = false;
        hitReport.materialIndex = triangle.materialIndex;
        cachedColor = triangle.color;
        useCachedColor = true;
    }

    return true;
}

float intersectAABB(const Ray ray, const vec3 invRayDir, const vec3 aabbMin, const vec3 aabbMax, float distance) {
    vec3 t1 = (aabbMin - ray.origin) * invRayDir;
    vec3 t2 = (aabbMax - ray.origin) * invRayDir;
    float tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    float tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    if (tmax >= tmin && tmin < distance && tmax > 0)
        return tmin;
    return 1000000.0;
}

void intersectBvh(const Ray ray, inout HitReport hitReport, uint root, uint meshIndex) {
    uint stack[64];
    uint stackPointer = 0;
    stack[stackPointer++] = root;

    while (stackPointer > 0) {
        uint nodeIndex = stack[--stackPointer];
        BVHNode node;
        mat4 transform = mat4(1.0);
        if (meshIndex == 0) {
            node = mesh1BVH.nodes[nodeIndex];
            transform = mesh1.transform;
        } else if (meshIndex == 1) {
            node = mesh2BVH.nodes[nodeIndex];
            transform = mesh2.transform;
        } else if (meshIndex == 2) {
            node = mesh3BVH.nodes[nodeIndex];
            transform = mesh3.transform;
        } else if (meshIndex == 3) {
            node = mesh4BVH.nodes[nodeIndex];
            transform = mesh4.transform;
        } else if (meshIndex == 4) {
            node = mesh5BVH.nodes[nodeIndex];
            transform = mesh5.transform;
        }
        vec3 invRayDir = 1.0 / ray.direction;

        vec3 min = (transform * vec4(node.min, 1.0)).xyz;
        vec3 max = (transform * vec4(node.max, 1.0)).xyz;
        if (intersectAABB(ray, invRayDir, min, max, hitReport.distance) == 1000000.0)
            continue;

        if (node.triangleCount > 0) {
            for (uint i = 0; i < node.triangleCount; i++) {
                uint triangleIndex = 0;
                if (meshIndex == 0) {
                    triangleIndex = mesh1PrimIndices.primIndices[node.leftChildOrFirstTriangle + i];
                    testTriangleIntersection(ray, hitReport, triangleIndex, 0, true);
                } else if (meshIndex == 1) {
                    triangleIndex = mesh2PrimIndices.primIndices[node.leftChildOrFirstTriangle + i];
                    testTriangleIntersection(ray, hitReport, triangleIndex, 1, true);
                } else if (meshIndex == 2) {
                    triangleIndex = mesh3PrimIndices.primIndices[node.leftChildOrFirstTriangle + i];
                    testTriangleIntersection(ray, hitReport, triangleIndex, 2, true);
                } else if (meshIndex == 3) {
                    triangleIndex = mesh4PrimIndices.primIndices[node.leftChildOrFirstTriangle + i];
                    testTriangleIntersection(ray, hitReport, triangleIndex, 3, true);
                } else if (meshIndex == 4) {
                    triangleIndex = mesh5PrimIndices.primIndices[node.leftChildOrFirstTriangle + i];
                    testTriangleIntersection(ray, hitReport, triangleIndex, 4, true);
                }
            }
        } else {
            stack[stackPointer++] = node.leftChildOrFirstTriangle;
            stack[stackPointer++] = node.leftChildOrFirstTriangle + 1;
        }
    }
}

bool testSceneTrace(Ray ray, inout HitReport hitReport) {
    bool hit = false;
    for (uint i = 0; i < scene.sphereCount; i++)
        if (testSphereIntersection(ray, hitReport, i))
            hit = true;

    for (uint i = 0; i < scene.planeCount; i++)
        if (testPlaneIntersection(ray, hitReport, i))
            hit = true;

    for (uint i = 0; i < scene.triangleCount; i++)
        if (testTriangleIntersection(ray, hitReport, i, 0, false))
            hit = true;

    if (scene.meshCount > 0)
        intersectBvh(ray, hitReport, 0, 0);
    if (scene.meshCount > 1)
        intersectBvh(ray, hitReport, 0, 1);
    if (scene.meshCount > 2)
        intersectBvh(ray, hitReport, 0, 2);
    if (scene.meshCount > 3)
        intersectBvh(ray, hitReport, 0, 3);
    if (scene.meshCount > 4)
        intersectBvh(ray, hitReport, 0, 4);
    if (scene.meshCount > 5)
        intersectBvh(ray, hitReport, 0, 5);
    if (scene.meshCount > 6)
        intersectBvh(ray, hitReport, 0, 6);
    if (scene.meshCount > 7)
        intersectBvh(ray, hitReport, 0, 7);
    if (scene.meshCount > 8)
        intersectBvh(ray, hitReport, 0, 8);
    if (scene.meshCount > 9)
        intersectBvh(ray, hitReport, 0, 9);

    return hit;
}

uint wangHash(inout uint seed) {
    // + 20 social credit for using the chinese hash
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);

    return seed;
}

float randomFloat01(inout uint state) {
    return float(wangHash(state)) / 4294967296.0;
}

vec3 randomUnitVector(inout uint state) {
    float z = randomFloat01(state) * 2.0f - 1.0f;
    float a = randomFloat01(state) * TWO_PI;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

float fresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident, float f0, float f90) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float cosX = -dot(normal, incident);

    if (n1 > n2) {
        float n = n1 / n2;
        float sinT2 = n * n * (1.0 - cosX * cosX);
        if (sinT2 > 1.0)
            return f90;

        cosX = sqrt(1.0 - sinT2);
    }

    float x = 1.0 - cosX;
    float ret = r0 + (1.0 - r0) * x * x * x * x * x;
    return mix(f0, f90, ret);
}

vec3 lessThan(vec3 f, float value) {
    return vec3(
        (f.x < value) ? 1.0f : 0.0f,
        (f.y < value) ? 1.0f : 0.0f,
        (f.z < value) ? 1.0f : 0.0f
    );
}

vec3 linearToSRGB(vec3 rgb) {
    rgb = clamp(rgb, 0.0f, 1.0f);

    return mix(
        pow(rgb, vec3(1.0f / 2.4f)) * 1.055f - 0.055f, rgb * 12.92f, lessThan(rgb, 0.0031308f)
    );
}

vec3 SRGBToLinear(vec3 rgb) {
    rgb = clamp(rgb, 0.0f, 1.0f);

    return mix(
        pow(((rgb + 0.055f) / 1.055f), vec3(2.4f)),
        rgb / 12.92f,
        lessThan(rgb, 0.04045f)
	);
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0f, 1.0f);
}

vec3 raytrace(Ray ray, inout uint rngState) {
    vec3 ret = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (uint i = 0; i < MAX_BOUNCES; ++i) {
        HitReport hitReport;
        hitReport.distance = 1000000.0;
        hitReport.position = vec3(0.0);
        hitReport.normal = vec3(0.0);
        hitReport.fromInside = false;

        testSceneTrace(ray, hitReport);

        if (hitReport.distance == 1000000.0) {
            if (useSkybox) {
                // Sample the skybox based on the ray direction with spherical coordinates
                vec2 uv = vec2(atan(ray.direction.z, ray.direction.x) / TWO_PI + 0.5, asin(ray.direction.y) / PI + 0.5);
                ret += SRGBToLinear(texture(skybox, uv).xyz) * throughput;
                return ret;
            } else {
                ret += throughput * vec3(0.0);
                return ret;
            }
        }

        Material material = scene.materials[hitReport.materialIndex];
        if (useCachedColor)
            material.albedo = cachedColor;
        useCachedColor = false;

        if (hitReport.fromInside)
            throughput *= exp(-material.refractiveColor * hitReport.distance);

        // Select if we should do a specular or diffuse reflection based on the fresnel reflectance of the material
        float specularChance = material.indirectSpecular;
        float refractionChance = material.refractionChance;

        // Apply the fresnel reflectance to the energy
        float rayProbability = 1.0f;
        if (specularChance > 0) {
            specularChance = fresnelReflectAmount(
                hitReport.fromInside ? material.refractiveIndex : 1.0,
                !hitReport.fromInside ? material.refractiveIndex : 1.0,
                ray.direction, hitReport.normal, material.indirectSpecular, 1.0f);

            float chanceMultiplier = (1.0f - specularChance) / (1.0f - material.indirectSpecular);
            refractionChance *= chanceMultiplier;
        }


        // Choose between specular, diffuse or refractive reflection
        float doSpecular = 0.0;
        float doRefraction = 0.0;
        float raySelectRoll = randomFloat01(rngState);
        if (specularChance > 0.0f && raySelectRoll < specularChance) {
            doSpecular = 1.0f;
            rayProbability = specularChance;
        } else if (refractionChance > 0.0f && raySelectRoll < specularChance + refractionChance) {
            doRefraction = 1.0f;
            rayProbability = refractionChance;
        } else {
            rayProbability = 1.0f - (specularChance + refractionChance);
        }


        rayProbability = max(rayProbability, 0.0001);


        // Set the new pos of the ray
        if (doRefraction == 1.0)
            ray.origin = hitReport.position - hitReport.normal * 0.001;
        else
            ray.origin = hitReport.position + hitReport.normal * 0.001;


        // Select the new direction of the ray
        vec3 diffuseRayDir = normalize(hitReport.normal + randomUnitVector(rngState));

        vec3 specularRayDir = reflect(ray.direction, hitReport.normal);
        specularRayDir = normalize(mix(specularRayDir, diffuseRayDir, material.specularRoughness * material.specularRoughness));

        vec3 refractionRayDir = refract(ray.direction, hitReport.normal, hitReport.fromInside ? material.refractiveIndex : 1.0f / material.refractiveIndex);
        refractionRayDir = normalize(mix(refractionRayDir, normalize(-hitReport.normal + randomUnitVector(rngState)), material.refractiveRoughness * material.refractiveRoughness));

        ray.direction = mix(diffuseRayDir, specularRayDir, doSpecular);
        ray.direction = mix(ray.direction, refractionRayDir, doRefraction);


        // Calculate the energy of the ray
        ret += throughput * material.emissive;
        if (doRefraction == 0.0)
            throughput *= mix(material.albedo, material.specularColor, doSpecular);

        throughput /= rayProbability;


        {   // Russian roulette
            /*
                The probability of continuing the path is the maximum of the color components.
                If a random number is greater than this probability, the path is terminated.
                Otherwise, the throughput is divided by the probability.

                This is a way to avoid paths that have very low throughput.
            */

            float p = max(throughput.r, max(throughput.g, throughput.b));
            if (randomFloat01(rngState) > p)
                break;

            throughput *= 1.0 / p;
        }
    }

    return ret;
}

void main() {
    vec4 target = camera.inverseProjection * vec4(inTexCoords * 2.0 - 1.0, -1.0, 1.0);

    // Add jitter to the ray direction to perform antialiasing
    // The jitter will not be visible in the final image because the accumulation will average the results
    uint rngState = uint(uint(gl_FragCoord.x) * uint(1973) + uint(gl_FragCoord.y) * uint(9277) + uint(seed) * uint(26699)) | uint(1);
    vec2 jitter = vec2(randomFloat01(rngState), randomFloat01(rngState)) - 0.5;
    target.xy += jitter / camera.resolution;

    // Calculate the ray direction
    vec3 projectedRayDir = vec3(target.x, target.y, target.z) / target.w;
    vec4 rayDir = camera.inverseView * vec4(projectedRayDir.x, projectedRayDir.y, projectedRayDir.z, 0);

    Ray ray;
    ray.origin = camera.position;
    ray.direction = normalize(rayDir.xyz);
    ray.energy = vec3(1.0);

    vec3 color = vec3(0.0);
    // Perform multiple samples per pixel to reduce noise and perform antialiasing with the jitter
    if (debugView == 0) {
        for (uint i = 0; i < SAMPLES; ++i)
            color += raytrace(ray, rngState) / float(SAMPLES);
        color = ACESFilm(color);
        color = linearToSRGB(color);
    } else {
        HitReport hitReport;
        hitReport.distance = 1000000.0;
        hitReport.position = vec3(0.0);
        hitReport.normal = vec3(0.0);
        hitReport.fromInside = false;

        testSceneTrace(ray, hitReport);

        if (hitReport.distance != 1000000.0) {
            if (debugView == 1) // albedo
                color = scene.materials[hitReport.materialIndex].albedo;
            else if (debugView == 2) // normal
                color = hitReport.normal * 0.5 + 0.5;
            else if (debugView == 3) // from inside
                color = vec3(hitReport.fromInside);
        }
    }

    outColor = vec4(color, 1.0);
}
