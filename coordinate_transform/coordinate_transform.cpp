#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
using namespace std;

struct Vec3 {
    double x, y, z;

    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
};

struct Quaternion {
    double w, x, y, z;

    Quaternion(double w = 1, double x = 0, double y = 0, double z = 0)
        : w(w), x(x), y(y), z(z) {}

    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        );
    }

    void normalize() {
        double n = sqrt(w * w + x * x + y * y + z * z);
        w /= n;
        x /= n;
        y /= n;
        z /= n;
    }

    Quaternion inverse() const {
        return Quaternion(w, -x, -y, -z);
    }

    Vec3 rotate(const Vec3& v) const {
        Quaternion p(0, v.x, v.y, v.z);
        Quaternion r = (*this) * p * inverse();
        return Vec3(r.x, r.y, r.z);
    }

    static Quaternion fromEuler(double yaw, double pitch, double roll) {
        double cy = cos(yaw / 2), sy = sin(yaw / 2);
        double cp = cos(pitch / 2), sp = sin(pitch / 2);
        double cr = cos(roll / 2), sr = sin(roll / 2);

        Quaternion q(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );

        q.normalize();
        return q;
    }

    void toEuler(double& yaw, double& pitch, double& roll) {
        normalize();

        roll = atan2(
            2 * (w * x + y * z),
            1 - 2 * (x * x + y * y)
        );

        double s = 2 * (w * y - z * x);
        if (fabs(s) >= 1)
            pitch = copysign(M_PI / 2, s);
        else
            pitch = asin(s);

        yaw = atan2(
            2 * (w * z + x * y),
            1 - 2 * (y * y + z * z)
        );
    }
};

struct Pose {
    Vec3 p;
    double yaw, pitch, roll;

    Pose(double x = 0, double y = 0, double z = 0,
         double yaw = 0, double pitch = 0, double roll = 0)
        : p(x, y, z), yaw(yaw), pitch(pitch), roll(roll) {}

    Quaternion q() const {
        return Quaternion::fromEuler(yaw, pitch, roll);
    }
};

struct Transform {
    Vec3 t;
    Quaternion q;

    Transform(Vec3 t = Vec3(), Quaternion q = Quaternion())
        : t(t), q(q) {}
};

Pose transformPose(const Pose& pose, const Transform& tf) {
    Vec3 newP = tf.q.rotate(pose.p) + tf.t;

    Quaternion newQ = tf.q * pose.q();
    newQ.normalize();

    double yaw, pitch, roll;
    newQ.toEuler(yaw, pitch, roll);

    return Pose(newP.x, newP.y, newP.z, yaw, pitch, roll);
}

int main() {
    double x, y, z, yaw, pitch, roll;
    string a, b, target;

    cin >> x >> y >> z >> yaw >> pitch >> roll;
    cin >> a >> b >> target;

    Pose cameraPose(x, y, z, yaw, pitch, roll);

    Transform cameraToGimbal(
        Vec3(2, 0, 0),
        Quaternion::fromEuler(0, 0, 0)
    );

    Transform gimbalToOdom(
        Vec3(0, 0, 0),
        Quaternion::fromEuler(-0.1, -0.1, -0.1)
    );

    Pose ans;

    if (target == "/Camera") {
        ans = cameraPose;
    } else if (target == "/Gimbal") {
        ans = transformPose(cameraPose, cameraToGimbal);
    } else if (target == "/Odom") {
        Pose gimbalPose = transformPose(cameraPose, cameraToGimbal);
        ans = transformPose(gimbalPose, gimbalToOdom);
    } else {
        return 0;
    }
    cout << fixed << setprecision(2)
         << ans.p.x << " " << ans.p.y << " " << ans.p.z << " "
         << ans.yaw << " " << ans.pitch << " " << ans.roll << endl;

    return 0;
}