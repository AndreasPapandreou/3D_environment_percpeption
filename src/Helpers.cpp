#include "Helpers.h"

#define CARLA_DATA 0
#define LYFT_DATA 1

double Helpers::dotProduct(const vec& v1, const vec& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

void Helpers::getBoxes(VecArray& borders, std::string& pathToVehicles){
    /// get size of binary
    std::ifstream binaryFile2;
    std::ostringstream oss;
    oss << pathToVehicles;
    std::string name = oss.str();

    binaryFile2.open(name, std::ios::in | std::ios::binary);
    binaryFile2.seekg(0, std::ios::end);
    int size = binaryFile2.tellg();

    std::ifstream binaryFile;
    binaryFile.seekg(0, std::ios::beg);
    binaryFile.open(name, std::ios::in | std::ios::binary);

    int index2{0}; float f;
    int num{0};
//    std::cout << "loading data...\n";
    vec p;
    while(binaryFile.tellg() < size)
    {
        binaryFile.read((char*)(&f), sizeof(float));
        p.x = f;
        binaryFile.read((char*)(&f), sizeof(float));
        p.y = f;
        binaryFile.read((char*)(&f), sizeof(float));
        p.z = f;

        borders.emplace_back(p);

        /// there are x,y,z values
        index2 += 3;
        num += 3 * sizeof(float);
        binaryFile.seekg(num, std::ios::beg);
    }
    binaryFile.close();
}

///    Checks whether points are inside the box.
///    Picks one corner as reference (p2) and computes the vector to a target point (v).
///    Then for each of the 3 axes, project v onto the axis and compare the length.
///    Inspired by: https://math.stackexchange.com/a/1552579
bool Helpers::pointInBox(const vec& p_check, const VecArray& corners) {
    vec p_ref{corners[6]}, p_x{corners[7]}, p_y{corners[5]}, p_z{corners[2]};

    vec i, j, k, v;
    i = p_x - p_ref;
    j = p_y - p_ref;
    k = p_z - p_ref;
    v = p_check - p_ref;

    double iv, jv, kv;
    iv = dotProduct(i, v);
    jv = dotProduct(j, v);
    kv = dotProduct(k, v);

    return (0 <= iv && iv <= dotProduct(i, i)) &&
           (0 <= jv && jv <= dotProduct(j, j)) &&
           (0 <= kv && kv <= dotProduct(k, k));
}

vec Helpers::getCenter(const VecArray& data) {
    vec center{0.0f, 0.0f, 0.0f};
    for (auto &p : data)
        center += p;
    center /= float(data.size());
    return center;
}

int Helpers::existInVector(std::vector<types>& data, const std::string& value) {
    for (auto &p : data)
        if (p.name == value)
            return p.index;
    return -1;
}

std::string Helpers::getCurrentDir() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        return cwd;
    }
}

//SphericalCoordinates Helpers::convertToSperical(vec point) {
//    SphericalCoordinates sc;
//    sc.r = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
//    sc.th = atan(point.y/point.x) * 180.0 / M_PI; /// convert radian to degree
//    sc.f = acos(point.z/sc.r) * 180.0 / M_PI; /// convert radian to degree
//    return sc;
//}

/// project lidar data to image based on https://github.com/windowsub0406/KITTI_Tutorial/blob/master/velo2cam_projection.ipynb
/// general pipeline :
/// https://codeyarns.github.io/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
/// image cordinates = normalized_image_coordinates x camera_coordinates x velodine_coordinates
void Helpers::projectToIm(const VecArray& lidar, VecArray& camera, calibMat& calib, vec& egoLocation, float yaw, float pitch, float roll) {
    /// project lidar data to camera
    Eigen::VectorXd res(3,1), eachLidar(4,1);
    Eigen::MatrixXd M1(3,3), M2(3,4), M3(4,4);

    /// from http://ftp.cs.toronto.edu/pub/psala/VM/camera-parameters.pdf
    /// M1(1,1) = WIDTH/(2*math.tan((focal_length/lens_x_size)*math.pi/360.0))

    #if CARLA_DATA == 1
        M1 << 859.238, 0.0, 1024.0,
          0.0, 859.238, 512.0,
          0.0, 0.0, 1.0;

        M2 << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0;

        M3 << 0.0, -1.0, 0.0, 0.0,
              0.0, 0.0, -1.0, 0.0,
              1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0;
    #endif

    /// lyft, sensors specs from https://medium.com/machine-learning-engineer/3-self-driving-car-dataset-for-deep-learning-research-3f44e5f67414
    #if LYFT_DATA == 1
        M1 << 886.818, 0.0, 612.0,
              0.0, 874.027, 512.0,
              0.0, 0.0, 1.0;

        M2 << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0;

//    "rotation": [0.024428220672382876, -0.011341011038338357, 0.00027714126227969363, 0.9996372175425093],
//    "translation": [1.1970994875774537, 0.0001418448487894911, 1.8279670539933053]

//    M3 << 0.0, -1.0, 0.0, 0.0,
//            0.0, 0.0, -1.0, 0.0,
//            1.0, 0.0, 0.0, 0.0,
//            0.0, 0.0, 0.0, 1.0;

//        M3 << 0.0, -1.0, 0.0, 1.197, ///
//              0.0, 0.0, -1.0, 0.00014, /// εμπρος-πισω, πισω -> (+) 0.5 -> 0.2
//              1.0, 0.0, 0.0, 1.8279, /// left-right, deksia -> (+)
//              0.0, 0.0, 0.0, 1.0;

        M3 << 0.0, -1.0, 0.0, 0.0, /// αριστερα δεξια
              0.0, 0.0, 1.0, 0.0, /// εμπρος-πισω
              -1.0, 0.0, 0.0, 0.0, ///
              0.0, 0.0, 0.0, 1.0;

//        M3 << 0.0, -1.0, 0.0, 1.197, ///
//              0.0, 0.0, -1.0, -0.00014, /// εμπρος-πισω, πισω -> (+) 0.5 -> 0.2
//              1.0, 0.0, 0.0, 1.8279, /// left-right, deksia -> (+)
//              0.0, 0.0, 0.0, 1.0;


//        yaw = (yaw * (M_PI / 180)); /// convert degree to radian
//        pitch = (pitch * (M_PI / 180)); /// convert degree to radian
//        roll = (roll * (M_PI / 180)); /// convert degree to radian
////
//        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
//        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
//        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
//        Eigen::Quaternion<double> q =  yawAngle*pitchAngle *rollAngle;
//        Eigen::Matrix3d rotationMatrix = q.matrix();

//        M3(0,0) = rotationMatrix(0,0);
//        M3(0,1) = rotationMatrix(0,1);
//        M3(0,2) = rotationMatrix(0,2);
//
//        M3(1,0) = rotationMatrix(1,0);
//        M3(1,1) = rotationMatrix(1,1);
//        M3(1,2) = rotationMatrix(1,2);
//
//        M3(2,0) = rotationMatrix(2,0);
//        M3(2,1) = rotationMatrix(2,1);
//        M3(2,2) = rotationMatrix(2,2);
//
//        M3(0,3) = 1.197;
//        M3(1,3) = 0.00014;
//        M3(2,3) = 1.8279;
//
//        M3(3,0) = 0.0;
//        M3(3,1) = 0.0;
//        M3(3,2) = 0.0;
//        M3(3,3) = 1.0;

//        std::cout << "M3 = " << M3 << std::endl;

#endif

    for (auto &l : lidar) {
        /// add threshold in order not to projevt points that belong behind the camera
        eachLidar(0,0) = l.x - egoLocation.x;
        eachLidar(1,0) = l.y - egoLocation.y;
        eachLidar(2,0) = l.z - egoLocation.z;
        eachLidar(3,0) = 1.0;

        res = M1*(M2*(M3*eachLidar));

        res(0,0) = res(0,0)/res(2,0);
        res(1,0) = res(1,0)/res(2,0);

        /// remove pixels falling out of image plane
        if (res(0,0) >=0 && res(1,0) >= 0) {
            camera.emplace_back(vec(abs(res(0,0)), abs(res(1,0)), 1.0));
        }
    }
}

void Helpers::rotate_data(VecArray &points, float angle_radian, glm::vec3 myRotationAxis) {
    glm::mat4 rotation =  glm::rotate(angle_radian, myRotationAxis);
    glm::vec3 new_p;
    for (auto&p : points) {
        new_p.x = p.x; new_p.y = p.y; new_p.z = p.z;
        new_p = glm::mat3(rotation) * new_p;
        p.x = new_p.x; p.y = new_p.y; p.z = new_p.z;
    }
}

void Helpers::rotate_data(vec &p, float angle_radian, glm::vec3 myRotationAxis) {
    glm::mat4 rotation =  glm::rotate(angle_radian, myRotationAxis);
    glm::vec3 new_p;
    new_p.x = p.x; new_p.y = p.y; new_p.z = p.z;
    new_p = glm::mat3(rotation) * new_p;
    p.x = new_p.x; p.y = new_p.y; p.z = new_p.z;
}