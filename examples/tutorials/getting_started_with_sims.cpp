// getting_started_with_sims.cpp
// Implementation of "Getting started with NMR simulations" tutorial
// Describing a single quantum system
// https://github.com/IlyaKuprov/Spinach/blob/main/examples/tutorials/getting_started_with_sims.pdf

#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <assert.h>
#include <fstream>
#include "TGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TApplication.h"

using namespace std::complex_literals;

// Commutator function [A, B] = AB - BA
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
commutator(const Eigen::MatrixBase<Derived>& A, const Eigen::MatrixBase<Derived>& B) {
    return A * B - B * A;
}

// Function to compute the Kronecker product
template <typename MatrixType>
Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
kron(const MatrixType& A, const MatrixType& B) {
    // Get dimensions of input matrices
    int rowsA = A.rows();
    int colsA = A.cols();
    int rowsB = B.rows();
    int colsB = B.cols();

    // Initialize the resulting matrix
    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rowsA * rowsB, colsA * colsB);

    // Compute the Kronecker product
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
        }
    }

    return result;
}

// FFTShift for Eigen::VectorXcd
Eigen::VectorXcd fftshift(const Eigen::VectorXcd& input) {
    auto N = input.size();           // Length of the input vector
    auto half = N / 2;               // Midpoint index

    Eigen::VectorXcd shifted(N);    // Output vector

    if (N % 2 == 0) {
        // Even length: split exactly in half
        shifted.head(half) = input.tail(half);
        shifted.tail(half) = input.head(half);
    } else {
        // Odd length: shift by floor(N/2)
        shifted.head(half + 1) = input.tail(half + 1);
        shifted.tail(half) = input.head(half);
    }

    return shifted;
}

int main() {
    std::cout << "Describing a single quantum system...\n";

    // Define Pauli matrices
    Eigen::Matrix<std::complex<double>, 2, 2> sigmaX;
    // Pauli-X (σx)
    sigmaX(0, 0) = 0.0;
    sigmaX(0, 1) = 0.5;
    sigmaX(1, 0) = 0.5;
    sigmaX(1, 1) = 0.0;

    Eigen::Matrix<std::complex<double>, 2, 2> sigmaY;
    // Pauli-Y (σy)
    sigmaY(0, 0) = 0.0;
    sigmaY(0, 1) = -0.5i;
    sigmaY(1, 0) = 0.5i;
    sigmaY(1, 1) = 0.0;

    Eigen::Matrix<std::complex<double>, 2, 2> sigmaZ;
    // Pauli-Z (σz)
    sigmaZ(0, 0) = 0.5;
    sigmaZ(0, 1) = 0.0;
    sigmaZ(1, 0) = 0.0;
    sigmaZ(1, 1) = -0.5;

    Eigen::Matrix<std::complex<double>, 2, 2> unit;
    unit(0, 0) = 1.0;
    unit(0, 1) = 0.0;
    unit(1, 0) = 0.0;
    unit(1, 1) = 1.0;

    // Check the commutation relations
    assert(commutator(sigmaX, sigmaY) == (1.0i * sigmaZ));
    assert(commutator(sigmaZ, sigmaX) == (1.0i * sigmaY));
    assert(commutator(sigmaY, sigmaZ) == (1.0i * sigmaX));

    // Build two-spin operators
    auto Lx = kron(sigmaX, unit);
    auto Sx = kron(unit, sigmaX);

    auto Ly = kron(sigmaY, unit);
    auto Sy = kron(unit, sigmaY);

    auto Lz = kron(sigmaZ, unit);
    auto Sz = kron(unit, sigmaZ);

    auto omega_L = 2 * M_PI * 200;
    auto omega_S = 2 * M_PI * 400;
    auto omega_J = 2 * M_PI * 40;

    auto hL = omega_L * Lz;
    auto hS = omega_S * Sz;
    auto hJ = omega_J * (Lx * Sx) + (Ly * Sy) + (Lz * Sz);

    std::cout << "Building Hamiltonian and calculating its 2-norm...\n";
    // Build the Hamiltonian
    auto H = hL + hS + hJ;
    // Calculate 2-norm to calculate time step
    auto H_eval = H.eval();
    auto HT = H.transpose();
    auto HTH = (HT * H_eval).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, 4, 4>> eigensolver(HTH);
    // The largest eigenvalue's square root is the 2-norm
    auto H_norm = std::sqrt(eigensolver.eigenvalues().maxCoeff());

    // Initial and detection state
    // initial state
    auto rho = Lx + Sx;
    // detection state
    auto detectionX = Lx + Sx;
    auto detectionY = (Ly + Sy) * 1.0i;
    auto coil = detectionX + detectionY;

    // Decide the time step
    // 2-norm is used here
    auto time_step = 0.5 / H_norm;

    // Build the step propagator
    std::complex<double> I(0.0, 1.0);  // Imaginary unit
    auto step_propagator = (-I * H * time_step).exp();

    // Build propagators aka Free Induction Decay (FID)
    // Time evolution with for 2048 steps
    std::cout << "Building propagator...\n";
    std::array<std::complex<double>, 2048> fid{};
    auto rho_calc = rho.eval();
    for (int i = 0; i < 2048; ++i) {
        auto rho_coil = rho_calc * coil;
        auto rho_coil_eval = rho_coil.eval();
        fid[i] = rho_coil_eval.trace();

        auto step_propagator_rho = step_propagator * rho_calc;
        auto step_propagator_adjoint = step_propagator.adjoint();
        auto rho_new = step_propagator_rho * step_propagator_adjoint;
        rho_calc = rho_new.eval();
    }

    // Apodization
    std::cout << "Apodization...\n";
    Eigen::ArrayXd linspace = Eigen::ArrayXd::LinSpaced(2048, 0, 1);
    auto window_function = (-5.0 * linspace).exp();
    auto window_function_eval = window_function.eval();

    std::array<std::complex<double>, 2048> fid_dot_star{};
    for (int i = 0; i < 2048; ++i) {
        fid_dot_star[i] = fid[i] * window_function_eval[i];
    }

    // DFT
    // Complex input signal
    std::cout << "Preparing input signal with zerofill...\n";
    Eigen::VectorXcd signal(8192);
    for (int i = 0; i < 2048; ++i) {
        signal[i] = fid_dot_star[i];
    }
    // with zerofill
    for (int i = 2048; i < 8192; ++i) {
        signal[i] = 0.0;
    }

    // exponentials
    std::cout << "Building matrix exponentials...\n";
    Eigen::MatrixXcd DFT(8192, 8192);
    for (int k = 0; k < 8192; ++k) {
        for (int n = 0; n < 8192; ++n) {
            DFT(k, n) = std::exp(-I * 2.0 * M_PI * (double(k * n) / 8192));
        }
    }

    // Compute Fourier Transform
    // Eigen::VectorXcd spectrum_not_shifted = DFT * signal;
    // Eigen::VectorXcd spectrum = fftshift(spectrum_not_shifted);
    // without fftshift
    std::cout << "Computing spectrum...\n";
    Eigen::VectorXcd spectrum = DFT * signal;

    std::cout << "Extracting frequencies and magnitudes...\n";
    // Splitting spectrum on frequencies and magnitudes
    double sampling_frequency = 1/time_step;
    auto N = spectrum.size();
    double delta_frequency = sampling_frequency / N;
    std::vector<double> frequencies;
    std::vector<double> magnitudes;

    // Compute frequencies and magnitudes (single-sided spectrum)
    for (int k = 0; k <= N / 2; ++k) {
        double frequency = k * delta_frequency;
        double magnitude = std::abs(spectrum[k]); // Magnitude is |z|

        frequencies.push_back(frequency);
        magnitudes.push_back(magnitude);
    }

    int argc{0};
    TApplication app("ROOT Application", &argc, nullptr);

    // Create a TGraph object to store the data points
    TGraph *graph = new TGraph(frequencies.size(), frequencies.data(), magnitudes.data());

    // Create a canvas to display the graph
    TCanvas *canvas = new TCanvas("canvas", "Frequency vs Magnitude", 800, 600);

    // Set graph title and axis labels
    graph->SetTitle("Frequency vs Magnitude");
    auto Xaxis = graph->GetXaxis();
    Xaxis->SetTitle("Frequency (Hz)");
    graph->GetYaxis()->SetTitle("Magnitude");

    std::cout << "Drawing frequencies and magnitudes...\n";
    // Draw the graph
    graph->Draw("ALP");  // "ALP" stands for Axis, Line, and Points (points connected by lines)

    std::cout << "Saving the drawing to png file...\n";
    // Save the plot to a file (optional)
    canvas->SaveAs("single_quantum_system_spectrum.png");

    // Output to csv
    std::cout << "Output frequencies and magnitudes to csv file...\n";
    std::ofstream output("./single_quantum_system_spectrum.csv");
    for (int k = 0; k <= N / 2; ++k) {
        output << frequencies[k] << "," << magnitudes[k] << std::endl;
    }
    output.close();

    app.Run();

    std::cout << "Describing a single quantum system completed.\n";

    return 0;
}
