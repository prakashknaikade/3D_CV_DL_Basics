# Photometric Stereo Overview

Photometric Stereo involves capturing multiple images of a static scene from a fixed viewpoint under varying illumination directions. By analyzing the changes in shading across these images, it is possible to recover surface normals and albedo.

---

## Reasons for Choosing Photometric Stereo

1. **Known Geometry and Normals**  
   - The plane's geometry is provided, and the surface normals are available from the normal maps.  
   - This simplifies the problem, focusing on albedo estimation.

2. **Controlled Lighting Conditions**  
   - Images are captured with one light turned on at a time, and the positions of the lights are known.  
   - This setup aligns perfectly with photometric stereo requirements, which depend on knowledge of light directions.

3. **Lambertian Surface Assumption**  
   - The wooden plane is primarily a diffuse reflector with known material properties:
     - Metallic: `0.0`
     - Specular color: `0.04`
     - Roughness: `0.3`
   - While there is some specular reflection, the dominant component is diffuse, fitting well with the Lambertian reflectance model used in photometric stereo.

---

## Implementation Details

### Equation Used

The fundamental equation for photometric stereo is:

Iᵢ = ρ (Lᵢ ⋅ N)

Where:  

- **Iᵢ**: Observed intensity under the *i*-th light source.
- **ρ**: Albedo (reflectance) of the surface.  
- **Lᵢ**: Light direction vector.  
- **N**: Surface normal vector.

---

## Advantages

1. **Simplicity and Efficiency**  
   - Photometric stereo provides a straightforward approach to estimate albedo without relying on complex optimization or machine learning models.

2. **Utilization of Provided Data**  
   - The methodology effectively leverages the provided images, calibration data, and known material properties.

---

## Limitations

1. **Diffuse Reflection Only**  
   - The Lambertian model assumes that the surface reflects light uniformly in all directions (purely diffuse).  
   - Real-world materials, including wood, exhibit both diffuse and specular reflections.

2. **Specular Reflections**  
   - Specular highlights violate the Lambertian assumption, leading to errors in albedo estimation.  
   - Thresholding helps mitigate this, but it may not completely eliminate the influence of specular reflections.

3. **Advanced Models Needed**  
    - More sophisticated methods, such as separating diffuse and specular components using the dichromatic reflection model, could provide better results but increase complexity.
