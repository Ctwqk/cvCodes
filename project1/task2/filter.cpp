/*
Author: Taiwei Cui
TIme: 2024-1-22

Portfolio upgrade: the filters keep their original behaviour and signatures
(const-correct now), with these fixes:
 - myRGB2Gray computed the grey image and then returned the blue channel instead;
 - sepia printed debug text on every call and accumulated into uchar;
 - blur5x5_1 / auxCalFunc accumulated into 8-bit values, losing up to a dozen
   grey levels per pixel; both blurs now round correctly and agree exactly;
 - the Sobel passes copied raw pixels into the border instead of zeros;
 - magnitude() wrapped around above 255 instead of saturating;
 - blurQuantize used `levels` as a bucket size, not a number of levels.
*/
#include "filter.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {
bool isBGR8(const cv::Mat& m) { return !m.empty() && m.type() == CV_8UC3; }
}  // namespace

// ---------------------------------------------------------------------------
// Greyscale / colour effects
// ---------------------------------------------------------------------------

// The grey level is 255 minus the average of the three channels.
int myRGB2Gray(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    cv::Mat out(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        const uchar* s = src.ptr<uchar>(i);
        uchar* o = out.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            const int sum = s[3 * j] + s[3 * j + 1] + s[3 * j + 2];
            o[j] = static_cast<uchar>(255 - (sum + 1) / 3);
        }
    }
    dst = out;
    return 0;
}

// Sepia filter; the edges are darkened: the further a pixel is from the
// centre, the darker it gets (1 - (dx^4 + dy^4) / 2 with dx, dy in [-1, 1]).
int sepia(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    static const double sepiaMat[3][3] = {{0.272, 0.534, 0.131},
                                          {0.349, 0.686, 0.168},
                                          {0.393, 0.769, 0.189}};
    const int row = src.rows, col = src.cols;
    cv::Mat out(row, col, CV_8UC3);
    for (int i = 0; i < row; i++) {
        const uchar* s = src.ptr<uchar>(i);
        uchar* o = out.ptr<uchar>(i);
        const double dy = 2.0 * i / row - 1.0;
        for (int j = 0; j < col; j++) {
            const double dx = 2.0 * j / col - 1.0;
            const double vignette = 1.0 - (dx * dx * dx * dx + dy * dy * dy * dy) / 2.0;
            for (int k = 0; k < 3; k++) {
                const double v = s[3 * j] * sepiaMat[k][0] + s[3 * j + 1] * sepiaMat[k][1] +
                                 s[3 * j + 2] * sepiaMat[k][2];
                o[3 * j + k] = cv::saturate_cast<uchar>(cv::saturate_cast<uchar>(v) * vignette);
            }
        }
    }
    dst = out;
    return 0;
}

// ---------------------------------------------------------------------------
// Blur
// ---------------------------------------------------------------------------

static const int kGauss1D[5] = {1, 2, 4, 2, 1};  // sums to 10

// Pixel-wise 5x5 Gaussian: every output pixel is the weighted sum of its 5x5 neighbourhood.
int blur5x5_1(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    const int row = src.rows, col = src.cols;
    cv::Mat out = src.clone();
    for (int i = 2; i < row - 2; i++) {
        uchar* o = out.ptr<uchar>(i);
        for (int j = 2; j < col - 2; j++) {
            for (int k = 0; k < 3; k++) {
                int sum = 0;
                for (int l = 0; l < 5; l++) {
                    const uchar* s = src.ptr<uchar>(i + l - 2);
                    for (int v = 0; v < 5; v++) {
                        sum += kGauss1D[l] * kGauss1D[v] * s[3 * (j + v - 2) + k];
                    }
                }
                o[3 * j + k] = static_cast<uchar>((sum + 50) / 100);
            }
        }
    }
    dst = out;
    return 0;
}

// Separable version: a vertical pass keeps the un-normalised sums (16 bit),
// the horizontal pass normalises once, so the result equals blur5x5_1 exactly.
int blur5x5_2(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    const int row = src.rows, col = src.cols;
    cv::Mat out = src.clone();
    if (row < 5 || col < 5) {
        dst = out;
        return 0;
    }
    cv::Mat mid(row, col, CV_16UC3);
    for (int i = 2; i < row - 2; i++) {  // vertical pass
        const uchar* s[5];
        for (int l = 0; l < 5; l++) s[l] = src.ptr<uchar>(i + l - 2);
        ushort* m = mid.ptr<ushort>(i);
        for (int x = 0; x < 3 * col; x++) {
            m[x] = static_cast<ushort>(s[0][x] + 2 * s[1][x] + 4 * s[2][x] + 2 * s[3][x] + s[4][x]);
        }
    }
    for (int i = 2; i < row - 2; i++) {  // horizontal pass
        const ushort* m = mid.ptr<ushort>(i);
        uchar* o = out.ptr<uchar>(i);
        for (int j = 2; j < col - 2; j++) {
            for (int k = 0; k < 3; k++) {
                const int sum = m[3 * (j - 2) + k] + 2 * m[3 * (j - 1) + k] + 4 * m[3 * j + k] +
                                2 * m[3 * (j + 1) + k] + m[3 * (j + 2) + k];
                o[3 * j + k] = static_cast<uchar>((sum + 50) / 100);
            }
        }
    }
    dst = out;
    return 0;
}

// ---------------------------------------------------------------------------
// Sobel
// ---------------------------------------------------------------------------

// X: smooth vertically with [1 2 1], then differentiate horizontally with [-1 0 1].
int sobelX3x3(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    const int row = src.rows, col = src.cols;
    cv::Mat out = cv::Mat::zeros(row, col, CV_16SC3);
    if (row < 3 || col < 3) {
        dst = out;
        return 0;
    }
    cv::Mat mid(row, col, CV_16SC3);
    for (int i = 1; i < row - 1; i++) {
        const uchar* a = src.ptr<uchar>(i - 1);
        const uchar* b = src.ptr<uchar>(i);
        const uchar* c = src.ptr<uchar>(i + 1);
        short* m = mid.ptr<short>(i);
        for (int x = 0; x < 3 * col; x++) m[x] = static_cast<short>(a[x] + 2 * b[x] + c[x]);
    }
    for (int i = 1; i < row - 1; i++) {
        const short* m = mid.ptr<short>(i);
        short* o = out.ptr<short>(i);
        for (int j = 1; j < col - 1; j++) {
            for (int k = 0; k < 3; k++) o[3 * j + k] = static_cast<short>(m[3 * (j + 1) + k] - m[3 * (j - 1) + k]);
        }
    }
    dst = out;
    return 0;
}

// Y: smooth horizontally with [1 2 1], then differentiate vertically with [1 0 -1]
// (row above minus row below).
int sobelY3x3(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    const int row = src.rows, col = src.cols;
    cv::Mat out = cv::Mat::zeros(row, col, CV_16SC3);
    if (row < 3 || col < 3) {
        dst = out;
        return 0;
    }
    cv::Mat mid(row, col, CV_16SC3);
    for (int i = 0; i < row; i++) {
        const uchar* s = src.ptr<uchar>(i);
        short* m = mid.ptr<short>(i);
        for (int j = 1; j < col - 1; j++) {
            for (int k = 0; k < 3; k++) {
                m[3 * j + k] = static_cast<short>(s[3 * (j - 1) + k] + 2 * s[3 * j + k] + s[3 * (j + 1) + k]);
            }
        }
    }
    for (int i = 1; i < row - 1; i++) {
        const short* up = mid.ptr<short>(i - 1);
        const short* down = mid.ptr<short>(i + 1);
        short* o = out.ptr<short>(i);
        for (int j = 1; j < col - 1; j++) {
            for (int k = 0; k < 3; k++) o[3 * j + k] = static_cast<short>(up[3 * j + k] - down[3 * j + k]);
        }
    }
    dst = out;
    return 0;
}

// Gradient magnitude of the two Sobel images.
int magnitude(const cv::Mat& sx, const cv::Mat& sy, cv::Mat& dst) {
    if (sx.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3 || sx.size() != sy.size()) return -1;
    cv::Mat out(sx.rows, sx.cols, CV_8UC3);
    for (int i = 0; i < sx.rows; i++) {
        const short* xp = sx.ptr<short>(i);
        const short* yp = sy.ptr<short>(i);
        uchar* o = out.ptr<uchar>(i);
        for (int x = 0; x < 3 * sx.cols; x++) {
            o[x] = cv::saturate_cast<uchar>(std::sqrt(static_cast<double>(xp[x]) * xp[x] +
                                                      static_cast<double>(yp[x]) * yp[x]));
        }
    }
    dst = out;
    return 0;
}

// Blur the image and then quantise each channel into `levels` levels.
int blurQuantize(const cv::Mat& src, cv::Mat& dst, int levels) {
    if (!isBGR8(src) || levels < 1 || levels > 255) return -1;
    cv::Mat out;
    blur5x5_2(src, out);
    for (int i = 0; i < out.rows; i++) {
        uchar* p = out.ptr<uchar>(i);
        for (int x = 0; x < 3 * out.cols; x++) {
            const int bin = p[x] * levels / 256;
            p[x] = levels == 1 ? 0 : static_cast<uchar>((bin * 255 + (levels - 1) / 2) / (levels - 1));
        }
    }
    dst = out;
    return 0;
}

// ---------------------------------------------------------------------------
// Face zoom, colour-vision simulation, max filter, contrast, kaleidoscope
// ---------------------------------------------------------------------------

// Enlarges the frame and puts the (largest) face in the centre, inspired by the iPad camera.
int centralize(cv::Mat& tar, cv::Rect& face) {
    if (tar.empty()) return -1;
    if (face.width >= tar.cols / 2 || face.height >= tar.rows / 2) return 0;
    face.x -= face.width / 2;
    if (face.x < 0) face.x = 0;
    face.y -= face.height / 2;
    if (face.y < 0) face.y = 0;
    face.width *= 2;
    if (face.x + face.width > tar.cols) face.x = tar.cols - face.width;
    face.height *= 2;
    if (face.y + face.height > tar.rows) face.y = tar.rows - face.height;
    cv::Mat crop = tar(face).clone();
    cv::resize(crop, tar, cv::Size(640, 480));
    return 0;
}

// Visualise a view that cannot tell red and green apart: the new blue channel
// is the average of red and green, the new red and green are the old blue.
int confuseRedGreen(const cv::Mat& src, cv::Mat& dst) {
    if (!isBGR8(src)) return -1;
    cv::Mat out(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* s = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* o = out.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            const uchar blue = s[j][0];
            o[j] = cv::Vec3b(static_cast<uchar>((s[j][1] + s[j][2]) / 2), blue, blue);
        }
    }
    dst = out;
    return 0;
}

// 11x11 max filter, done as two 1-D passes; the 5-pixel frame is copied from src.
int localMax(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.depth() != CV_8U) return -1;
    const int R = 5, row = src.rows, col = src.cols, ch = src.channels();
    cv::Mat out = src.clone();
    if (row <= 2 * R || col <= 2 * R) {
        dst = out;
        return 0;
    }
    cv::Mat mid = src.clone();  // vertical maxima
    for (int i = R; i < row - R; i++) {
        uchar* m = mid.ptr<uchar>(i);
        for (int x = 0; x < ch * col; x++) {
            uchar best = 0;
            for (int k = i - R; k <= i + R; k++) best = std::max(best, src.ptr<uchar>(k)[x]);
            m[x] = best;
        }
    }
    for (int i = R; i < row - R; i++) {
        const uchar* m = mid.ptr<uchar>(i);
        uchar* o = out.ptr<uchar>(i);
        for (int j = R; j < col - R; j++) {
            for (int c = 0; c < ch; c++) {
                uchar best = 0;
                for (int k = j - R; k <= j + R; k++) best = std::max(best, m[ch * k + c]);
                o[ch * j + c] = best;
            }
        }
    }
    dst = out;
    return 0;
}

int brightAndContrast(const cv::Mat& src, cv::Mat& dst, float alpha, int beta) {
    if (src.empty()) return -1;
    cv::Mat out;
    src.convertTo(out, -1, alpha, beta);
    dst = out;
    return 0;
}

// mask = identity, except that the lower half of the rows is flipped, so that
// mask * tile mirrors the top half onto the bottom half (mask^T does the same for columns).
cv::Mat makeKaleidoscopeMask(int size) {
    cv::Mat mask = cv::Mat::eye(size, size, CV_32F);
    for (int i = size / 2; i < size; i++) {
        float* p = mask.ptr<float>(i);
        p[i] = 0;
        p[size - 1 - i] = 1;
    }
    return mask;
}

// Each of the num x num tiles becomes a 4-fold mirror of its own top-left quadrant.
int kaleidoscope(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask, int num, int size) {
    if (!isBGR8(src) || src.cols != src.rows) return -1;
    if (num < 1 || size < 2 || num * size > src.cols) return -1;
    if (mask.rows != size || mask.cols != size || mask.type() != CV_32F) return -1;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Mat core, tile;
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            for (int z = 0; z < 3; z++) {
                const cv::Rect r(i * size, j * size, size, size);
                channels[z](r).convertTo(core, CV_32F);
                cv::gemm(mask, core, 1, cv::Mat(), 0, tile);
                cv::gemm(tile, mask.t(), 1, cv::Mat(), 0, tile);
                tile.convertTo(tile, CV_8U);
                tile.copyTo(channels[z](r));
            }
        }
    }
    cv::merge(channels, dst);
    return 0;
}
