// WSINDy for 2D vacuum Maxwell in (x, z, t), fields Ex Ey Ez Bx By Bz.
// compile:  clang++ -O3 -std=c++17 -I /opt/homebrew/opt/eigen/include/eigen3 wsindy_maxwell.cpp -o wsindy_maxwell
// run:    ./wsindy_maxwell maxwell_vacuum.bin [noise_ratio] [max_deriv] [quadratic 0/1] [out_prefix] [seed]
//         writes <out_prefix>_coefs.csv and <out_prefix>_fit.csv for plot-recovery.py
// (make the .bin first:  python export-bin.py)
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <map>
#include <random>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

using namespace std;
using namespace Eigen;

// set up
constexpr int p = 4, q = 4;              // test fn (s-a)^p (b-s)^q ; need p,q >= 3 for 2nd derivatives
// window length (grid points) and stride between window starts, per axis
int win_t = 21, win_x = 25, win_z = 25;
int str_t = 3,  str_x = 4,  str_z = 4;
int max_deriv = 1;                       // highest spatial derivative order in candidate library
bool use_quadratic = false;              // add products Ex*Ey, Bx*Bx, etc. to candidate library

const vector<string> FN = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};

// data
int nt, nx, nz;
double dt, dx, dz;                       // microns (E in units of c)
vector<vector<double>> F(6);             // F[f][(it*nx + ix)*nz + iz]

void load(const string& fn) {
    ifstream in(fn, ios::binary);
    if (!in) { cerr << "cannot open " << fn << "\n"; exit(1); }
    int32_t dims[3]; double h[3];
    in.read((char*)dims, sizeof dims); in.read((char*)h, sizeof h);
    nt = dims[0]; nx = dims[1]; nz = dims[2]; dt = h[0]; dx = h[1]; dz = h[2];
    for (auto& f : F) { f.resize((size_t)nt * nx * nz); in.read((char*)f.data(), f.size() * sizeof(double)); }
    if (!in) { cerr << "file too short\n"; exit(1); }
}

void add_noise(double ratio, unsigned seed) {           // inject artificial gaussian noise, std = ratio * RMS of each field
    if (ratio <= 0) return;
    mt19937 gen(seed);
    for (auto& f : F) {
        double ms = 0; for (double v : f) ms += v * v;
        normal_distribution<double> nd(0.0, ratio * sqrt(ms / f.size()));
        for (double& v : f) v += nd(gen);
    }
}

// test functions
struct Axis {
    int n, m, stride; double h;
    vector<int> starts;                  // first grid index of each window
    std::array<vector<double>, 3> d;          // d[k][i] = k-th derivative of phi at the window's i-th point
};

Axis make_axis(int n, double h, int m, int stride) {
    Axis A{n, m, stride, h, {}, {}};
    for (int s = 0; s + m <= n; s += stride) A.starts.push_back(s);
    double a = 0, b = (m - 1) * h;
    double C = 1.0 / (pow(p, p) * pow(q, q)) * pow((p + q) / (b - a), p + q);   // same C as heat code
    for (auto& v : A.d) v.assign(m, 0.0);
    for (int i = 0; i < m; i++) {
        double f = i * h - a, g = b - i * h;
        A.d[0][i] = C * pow(f, p) * pow(g, q);
        A.d[1][i] = C * (p * pow(f, p - 1) * pow(g, q) - q * pow(f, p) * pow(g, q - 1));
        A.d[2][i] = C * (p * (p - 1) * pow(f, p - 2) * pow(g, q) - 2.0 * p * q * pow(f, p - 1) * pow(g, q - 1)
                         + q * (q - 1) * pow(f, p) * pow(g, q - 2));
    }
    return A;
}
Axis AT, AX, AZ;

// weak-form
// return: for every window (jt,jx,jz), (-1)^(ot+ox+oz) * integral of (d^ot_t d^ox_x d^oz_z phi) * data.
// phi is separable -> do sum one one axis at a time
// row order: (jt*Nx + jx)*Nz + jz
VectorXd weak_column(const vector<double>& data, int ot, int ox, int oz) {
    int Nt = AT.starts.size(), Nx = AX.starts.size(), Nz = AZ.starts.size();
    const auto& wt = AT.d[ot]; const auto& wx = AX.d[ox]; const auto& wz = AZ.d[oz];
    vector<double> A((size_t)nt * nx * Nz), B((size_t)nt * Nx * Nz);
    for (int it = 0; it < nt; it++) for (int ix = 0; ix < nx; ix++) {
        const double* row = &data[((size_t)it * nx + ix) * nz];
        for (int jz = 0; jz < Nz; jz++) {
            double s = 0; const double* r = row + AZ.starts[jz];
            for (int c = 0; c < AZ.m; c++) s += wz[c] * r[c];
            A[((size_t)it * nx + ix) * Nz + jz] = s;
        }
    }
    for (int it = 0; it < nt; it++) for (int jx = 0; jx < Nx; jx++) for (int jz = 0; jz < Nz; jz++) {
        double s = 0;
        for (int b = 0; b < AX.m; b++) s += wx[b] * A[((size_t)it * nx + AX.starts[jx] + b) * Nz + jz];
        B[((size_t)it * Nx + jx) * Nz + jz] = s;
    }
    VectorXd out(Nt * Nx * Nz);
    double scale = ((ot + ox + oz) % 2 ? -1.0 : 1.0) * dt * dx * dz;
    for (int jt = 0; jt < Nt; jt++) for (int jx = 0; jx < Nx; jx++) for (int jz = 0; jz < Nz; jz++) {
        double s = 0;
        for (int a = 0; a < AT.m; a++) s += wt[a] * B[((size_t)(AT.starts[jt] + a) * Nx + jx) * Nz + jz];
        out((jt * Nx + jx) * Nz + jz) = scale * s;
    }
    return out;
}

// candidate library terms
struct Term { string name; int f1, f2, bx, bz; }; 

vector<Term> build_terms() {
    vector<Term> T;
    for (int f = 0; f < 6; f++)
        for (int bx = 0; bx <= max_deriv; bx++) for (int bz = 0; bx + bz <= max_deriv; bz++) {
            string nm = FN[f];
            if (bx + bz > 0) nm += "_" + string(bx, 'x') + string(bz, 'z');
            T.push_back({nm, f, -1, bx, bz});
        }
    if (use_quadratic)
        for (int f1 = 0; f1 < 6; f1++) for (int f2 = f1; f2 < 6; f2++)
            T.push_back({FN[f1] + "*" + FN[f2], f1, f2, 0, 0});
    return T;
}

MatrixXd build_library(const vector<Term>& T) {
    int rows = AT.starts.size() * AX.starts.size() * AZ.starts.size();
    MatrixXd G(rows, T.size());
    for (size_t j = 0; j < T.size(); j++) {
        if (T[j].f2 < 0) G.col(j) = weak_column(F[T[j].f1], 0, T[j].bx, T[j].bz);
        else {
            vector<double> prod(F[0].size());
            for (size_t i = 0; i < prod.size(); i++) prod[i] = F[T[j].f1][i] * F[T[j].f2][i];
            G.col(j) = weak_column(prod, 0, 0, 0);
        }
    }
    return G;
}

// MSTLS
// (with rank-safe solver: dB/dx + dB/dz = 0 and div E = 0 make some library columns nearly collinear)
MatrixXd sub(const MatrixXd& M, const vector<int>& idx) {  
    MatrixXd S(M.rows(), idx.size());
    for (size_t i = 0; i < idx.size(); i++) S.col(i) = M.col(idx[i]);
    return S;
}
VectorXd lsq(const MatrixXd& A, const VectorXd& b) { return A.completeOrthogonalDecomposition().solve(b); }

VectorXd mstls(const MatrixXd& Theta, const VectorXd& target, double lambda) {
    int n = Theta.cols();
    VectorXd L(n), U(n);
    for (int j = 0; j < n; j++) {
        double r = target.norm() / Theta.col(j).norm();
        L(j) = lambda * max(1.0, r);
        U(j) = (1.0 / lambda) * min(1.0, r);
    }
    vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
    while (!idx.empty()) {
        size_t prior = idx.size();
        VectorXd xi = lsq(sub(Theta, idx), target);
        vector<int> keep;
        for (size_t i = 0; i < idx.size(); i++)
            if (abs(xi(i)) >= L(idx[i]) && abs(xi(i)) <= U(idx[i])) keep.push_back(idx[i]);
        idx = keep;
        if (idx.size() == prior) break;
    }
    VectorXd w = VectorXd::Zero(n);
    if (idx.empty()) return w;
    VectorXd xs = lsq(sub(Theta, idx), target);
    for (size_t i = 0; i < idx.size(); i++) w(idx[i]) = xs(i);
    return w;
}

// choose lambda by scanning (Messenger & Bortz 2021: distance to the full least-squares)
// fit, measured through the library, plus a sparsity penalty)
VectorXd mstls_scan(const MatrixXd& G, const VectorXd& b, double& best_lambda) {
    VectorXd wstar = lsq(G, b);
    double denom = (G * wstar).norm();
    VectorXd best; double best_loss = 1e300;
    for (int k = 0; k < 60; k++) {
        double lam = pow(10.0, -4.0 + 4.0 * k / 59.0 * 0.95);          // 1e-4 ... ~0.6
        VectorXd w = mstls(G, b, lam);
        int nnz = 0; for (int i = 0; i < w.size(); i++) if (w(i) != 0) nnz++;
        if (nnz == 0) continue;
        double loss = (G * (w - wstar)).norm() / denom + double(nnz) / G.cols();
        if (loss < best_loss) { best_loss = loss; best = w; best_lambda = lam; }
    }
    if (best.size() == 0) best = VectorXd::Zero(G.cols());
    return best;
}

int main(int argc, char** argv) {
    string fn = argc > 1 ? argv[1] : "maxwell_vacuum.bin";
    double noise = argc > 2 ? atof(argv[2]) : 0.0;
    if (argc > 3) max_deriv = atoi(argv[3]);
    if (argc > 4) use_quadratic = atoi(argv[4]) != 0;
    string prefix = argc > 5 ? argv[5] : "wsindy";
    unsigned seed = argc > 6 ? (unsigned)atoi(argv[6]) : 12345u;

    load(fn);
    cout << "data: nt,nx,nz = " << nt << "," << nx << "," << nz << "  dt,dx,dz = " << dt << "," << dx << "," << dz << " um\n";
    add_noise(noise, seed);

    AT = make_axis(nt, dt, win_t, str_t);
    AX = make_axis(nx, dx, win_x, str_x);
    AZ = make_axis(nz, dz, win_z, str_z);
    auto terms = build_terms();
    cout << "windows: " << AT.starts.size() << " x " << AX.starts.size() << " x " << AZ.starts.size()
         << " = " << AT.starts.size() * AX.starts.size() * AZ.starts.size() << " rows,  library terms: " << terms.size()
         << ",  noise ratio " << noise << "\n";

    MatrixXd G = build_library(terms);

    // expected vacuum Maxwell (units: E/c, x and ct in the same length unit)
    map<string, map<string, double>> truth = {
        {"Ex", {{"By_z", -1}}},
        {"Ey", {{"Bx_z", 1}, {"Bz_x", -1}}},
        {"Ez", {{"By_x", 1}}},
        {"Bx", {{"Ey_z", 1}}},
        {"By", {{"Ez_x", 1}, {"Ex_z", -1}}},
        {"Bz", {{"Ey_x", -1}}}};

    vector<VectorXd> Wall(6), Ball(6), GWall(6); vector<double> lams(6), ress(6);   // saved for the csv output
    cout << fixed << setprecision(4);
    for (int f = 0; f < 6; f++) {
        VectorXd b = weak_column(F[f], 1, 0, 0);               // -(phi_t, u)  = (phi, u_t)
        double lam = 0;
        VectorXd w = mstls_scan(G, b, lam);
        cout << "\nd" << FN[f] << "/dt =";
        auto& tr = truth[FN[f]];
        int found = 0, spurious = 0; double maxerr = 0;
        for (size_t j = 0; j < terms.size(); j++) {
            if (w(j) == 0) continue;
            cout << (w(j) >= 0 ? " +" : " -") << abs(w(j)) << "*" << terms[j].name;
            if (tr.count(terms[j].name)) { found++; maxerr = max(maxerr, abs(w(j) - tr[terms[j].name])); }
            else spurious++;
        }
        double res = (G * w - b).norm() / b.norm();
        Wall[f] = w; Ball[f] = b; GWall[f] = G * w; lams[f] = lam; ress[f] = res;
        cout << "\n   lambda=" << scientific << setprecision(2) << lam << "  rel. residual=" << res << fixed << setprecision(4)
             << "   true terms found " << found << "/" << tr.size() << ", spurious " << spurious
             << ", max coeff error on true terms " << maxerr << "\n";
    }
    {   // results for plot-recovery.py
        ofstream oc(prefix + "_coefs.csv");
        oc << "target,term,coef,truth,lambda,relres\n" << scientific << setprecision(8);
        for (int f = 0; f < 6; f++) for (size_t j = 0; j < terms.size(); j++) {
            auto& tr = truth[FN[f]];
            oc << FN[f] << "," << terms[j].name << "," << Wall[f](j) << ","
               << (tr.count(terms[j].name) ? tr[terms[j].name] : 0.0) << "," << lams[f] << "," << ress[f] << "\n";
        }
        ofstream of(prefix + "_fit.csv");      // weak-form fit: b (= (phi,u_t)) and G*w for each window
        of << "row"; for (auto& n : FN) of << ",b_" << n << ",Gw_" << n; of << "\n" << scientific << setprecision(6);
        for (int r = 0; r < Ball[0].size(); r++) {
            of << r; for (int f = 0; f < 6; f++) of << "," << Ball[f](r) << "," << GWall[f](r); of << "\n";
        }
    }
    cout << "\nexpected:\n  dEx/dt = -By_z   dEy/dt = +Bx_z -Bz_x   dEz/dt = +By_x\n"
            "  dBx/dt = +Ey_z   dBy/dt = +Ez_x -Ex_z   dBz/dt = -Ey_x\n";
}