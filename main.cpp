#include <iostream>
#include <fstream>
#include <vector>

#include <math.h>

#include "Eigen.h"

#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
    // normal normalized, stored as 3 floats (no extra 4th component that is 1.0)
    Vector3f normal;
};

inline float sqrdDist(Vector4f p1, Vector4f p2)
{
	float x = p1(0) - p2(0);
	float y = p1(1) - p2(1);
	float z = p1(2) - p2(2);

	return x*x + y*y + z*z; 
}

// A cross two Vector4f vectors, ment to be useful when their last element always is 1
Vector3f cross(Vector4f v1, Vector4f v2)
{
    Vector3f v31 = v1.head<3>();
    Vector3f v32 = v2.head<3>();
    Vector3f normal = (v31).cross(v32);
    return normal;
}

bool WriteMesh(std::vector<Vertex>& vertices, unsigned int width, unsigned int height, const std::string& filename)
{
    // file format http://paulbourke.net/dataformats/obj/minobj.html
    
	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

    // Save vertecies: the position (point cloud and normal)
    for (auto const& v: vertices)
	{
        outFile << "v"   << " " << v.position(0) << " " << v.position(1) << " " << v.position(2) << " " << std::endl;
        outFile << "vn " << " " << v.normal(0)   << " " << v.normal(1)   << " " << v.normal(2)   << " " << std::endl;
	}

	// close file
	outFile.close();

	return true;
}


void SetVertices(std::vector<Vertex>& validVertecies, VirtualSensor sensor, float edgeThresholdSqrd)
{
    // get ptr to the current depth frame
    // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
    float* depthMap = sensor.GetDepth();
    // get ptr to the current color frame
    // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
    BYTE* colorMap = sensor.GetColorRGBX();
    
    // get depth intrinsics
    Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
    Matrix3f depthIntrinsicsInv = sensor.GetDepthIntrinsics().inverse();
    float fovX = depthIntrinsics(0, 0);
    float fovY = depthIntrinsics(1, 1);
    float cX = depthIntrinsics(0, 2);
    float cY = depthIntrinsics(1, 2);
    
    // compute inverse depth extrinsics
    Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();
    
    Matrix4f trajectory = sensor.GetTrajectory();
    Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();
    
    // TODO 1: back-projection
    // write result to the vertices array below, keep pixel ordering!
    // if the depth value at idx is invalid (MINF) write the following values to the vertices array
    // vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
    // vertices[idx].color = Vector4uc(0,0,0,0);
    // otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
    
    // Ie we are taking the points from the camera screen to how the points are in the real world,
    // buy the real world I mean a creating all the (x,y,z) points that is seen from the camera.
    
    unsigned int imageWidth = sensor.GetDepthImageWidth();
    unsigned int imageHeight = sensor.GetDepthImageHeight();
    
    Vertex* vertices = new Vertex[imageWidth * imageHeight];
    
    for (int i = 0; i < imageWidth * imageHeight; ++i)
    {
        int u = i % imageWidth; // from top left and rightwards, ie column
        int v = i / imageWidth; // from top left and downwards, ie row
        
        float d  = depthMap[i]; // depth value at (v, u)
        Vector4f pos;           // position (x,y,z) seen from the camera screen at (v, u),
                                // still not real world cordinates because have not incoorporated
                                // the rotation and translation of the camera. Will be done later, see after if, else.
        Vector4uc c;            // color value at (v, u)
        
        if (d == MINF)
        {
            pos = Vector4f(MINF, MINF, MINF, MINF);
            c = Vector4uc(0, 0, 0, 0);
        }
        else
        {
            // Calculation below corresponds to pos = IntrinsicInv * d(v, u) * [v, u, 1].T
            // where Instrinc holds 2D translation, 2D scaling and 2D Shear
            // of camera screen due to camera pinhole and position of screen (film)
            // Hence, a shortened version of what would happen in the matrix multiplication above.
            Vector4f posImage = {v, u, 1.0, 1.0};
            
            Matrix4f depthIntrinsicsInvExtended; depthIntrinsicsInvExtended.setIdentity();
            depthIntrinsicsInvExtended.block(0, 0, 3, 3) = depthIntrinsicsInv;
            
            //pos = depthExtrinsicsInv * depthIntrinsicsInvExtended * (d * posImage);

            pos(0) = (u - cX) * depthMap[i] / fovX;
            pos(1) = (v - cY) * depthMap[i] / fovY;
            pos(2) = depthMap[i];
            pos(3) = 1.0f;
            
            BYTE color = colorMap[i];
            
            c(0) = colorMap[i*4 + 0];
            c(1) = colorMap[i*4 + 1];
            c(2) = colorMap[i*4 + 2];;
            c(3) = 255;
        }
        
        // Calculation: Every point (x,y,z) as seen from the camera (called camera space),
        // has to be turn into real world space, ie global (x,y,z) coordinates.
        // This is done taking the into account the camera's rotation (ie its viewing vector)
        // and its translation (ie position in the real world/global frame).
        // This can all be incoorporated into the following equation
        // (x,y,z,1)_global = trajecteryInv * (x,y,z,1)_camera
        // where trajectory is a (4,4) matrix with
        // rotation R (3,3) matrix in top left and
        // translation t (3) vector in top right and
        // 1.0 in bottom right corner and
        // else zeros

        vertices[i].position = trajectoryInv * pos;

        //vertices[i].position = pos;
        
        vertices[i].color = c;
    }
    
    // valid vertices
    std::vector<bool> isValid(imageWidth*imageHeight);
    for (int i = 0; i < imageWidth * imageHeight; ++i)
    {
        Vertex v = vertices[i];
        if (isnan(v.position(0)) || isnan(v.position(1)) || isnan(v.position(2)) ||
            isinf(v.position(0)) || isinf(v.position(1)) || isinf(v.position(2)))
        {
            isValid[i] = false;
        }
        else
        {
            isValid[i] = true;
        }
    }
    
    // edge indices
    std::vector<unsigned int> edgeIndices;
    for (int u = 0; u < imageWidth; ++u)
    {
        unsigned int upperEdgeIdx = u;
        unsigned int lowerEdgeIdx = u + imageWidth * (imageHeight - 1);
        edgeIndices.push_back(upperEdgeIdx);
        edgeIndices.push_back(lowerEdgeIdx);
    }
    for (int v = 0; v < imageHeight; ++v)
    {
        unsigned int rightEdgeIdx = v * imageWidth;
        unsigned int leftEdgeIdx  = v * imageWidth + (imageWidth - 1);
        edgeIndices.push_back(rightEdgeIdx);
        edgeIndices.push_back(leftEdgeIdx);
    }
    
    // calculate vertex normals
    std::vector<bool> hasNormal(imageWidth*imageHeight);
    for (auto const& edgeIdx: edgeIndices)
        hasNormal[edgeIdx] = false; // no normals on the egdes of depth image
    for (int u = 1; u < imageWidth - 1; ++u)
    {
        for (int v = 1; v < imageHeight - 1; ++v)
        {
            int idx      = v * imageWidth + u;
            int rightIdx = v * imageWidth + (u + 1);
            int leftIdx  = v * imageWidth + (u - 1);
            int lowerIdx = (v + 1) * imageWidth + u;
            int upperIdx = (v - 1) * imageWidth + u;
            if (isValid[idx]      == false ||
                isValid[rightIdx] == false ||
                isValid[leftIdx]  == false ||
                isValid[lowerIdx] == false ||
                isValid[upperIdx] == false)  // then one or more of the vertices does not exist
            {
                hasNormal[idx] = false;
            }
            else // then all 5 points exist
            {
                Vector4f p      = vertices[idx].position;
                Vector4f pRight = vertices[rightIdx].position;
                Vector4f pLeft  = vertices[leftIdx].position;
                Vector4f pLower = vertices[lowerIdx].position;
                Vector4f pUpper = vertices[rightIdx].position;
                if (sqrdDist(p, pRight) >= edgeThresholdSqrd ||
                    sqrdDist(p, pLeft)  >= edgeThresholdSqrd ||
                    sqrdDist(p, pLower) >= edgeThresholdSqrd ||
                    sqrdDist(p, pUpper) >= edgeThresholdSqrd)    // then points are too far from each other
                {
                    hasNormal[idx] = false;
                }
                else // then normal makes sence and can be computed
                {
                    hasNormal[idx] = true;
                    Vector4f v1 = pRight - pLeft;
                    Vector4f v2 = pUpper - pLower;
                    Vector3f normal = cross(v1, v2).normalized();
                    vertices[idx].normal = normal;
                }
            }
        }
    }
    
    for (int i = 0; i < imageWidth * imageHeight; ++i)
    {
        if (hasNormal[i] == true)
            validVertecies.push_back(vertices[i]);
    }
    
    std::cout << "imageWidth * imageHeight: " << imageWidth * imageHeight << std::endl;
    std::cout << "validVertecies.size(): " << validVertecies.size() << std::endl;
    std::cout << "validVertecies[0].normal: \n" << validVertecies[0].normal << std::endl;
}



int main()
{
	std::string filenameIn = "/Users/isakrathestore/Documents/3D Scanning/Exercise 1/IN2354-Exercise1/data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

    // max sqrdDistance between points to in order to compute a normal between them
    float edgeThresholdSqrd = 0.05f*0.05f; // 1cm
    
    int i = 0;
	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
        std::cout << "Starting to work with frame: " << i << std::endl;
        
        // get global vertices of frame
        std::vector<Vertex> vertices;
        SetVertices(vertices, sensor, edgeThresholdSqrd);
        
        std::cout << "Got global vertices from frame: " << i << std::endl;
        
		// write mesh file
		std::stringstream ss;
		ss << "/Users/isakrathestore/Documents/3D Scanning/Exercise 1/IN2354-Exercise1/mesh/"
           << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".obj";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}
        
        std::cout << "Wrote to file for global vertices from frame: " << i++ << std::endl << std::endl;
	}

	return 0;
}
