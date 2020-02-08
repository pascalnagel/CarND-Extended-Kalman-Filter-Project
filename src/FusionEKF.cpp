#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * Finish initializing the FusionEKF.
   * Set the process and measurement noises
   */
  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;
    ekf_.P_ = MatrixXd(4, 4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_(0)*cos(measurement_pack.raw_measurements_(1)), //x = rho*cos(phi)
                 measurement_pack.raw_measurements_(0)*sin(measurement_pack.raw_measurements_(1)), //y = rho*sin(phi)
                 0,
                 0;
      // x_error = |dx/drho|*rho_error + |dx/dphi|*phi_error
      float x_error = abs(cos(measurement_pack.raw_measurements_(1)))*R_radar_(0,0)
                 + abs(sin(measurement_pack.raw_measurements_(1)))*measurement_pack.raw_measurements_(0)*R_radar_(1,1);
      // y_error = |dy/drho|*rho_error + |dy/dphi|*phi_error
      float y_error = abs(sin(measurement_pack.raw_measurements_(1)))*R_radar_(0,0)
                 + abs(cos(measurement_pack.raw_measurements_(1)))*measurement_pack.raw_measurements_(0)*R_radar_(1,1);
      ekf_.P_ << x_error, 0, 0, 0,
                 0, y_error, 0, 0,
                 0, 0, 100000, 0,
                 0, 0, 0, 100000;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_(0), 
                 measurement_pack.raw_measurements_(1), 
                 0, 
                 0;
      ekf_.P_ << R_laser_(0,0), 0, 0, 0,
                 0, R_laser_(1,1), 0, 0,
                 0, 0, 100000, 0,
                 0, 0, 0, 100000;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Using noise_ax = 9 and noise_ay = 9 for Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;


  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  ekf_.Q_ << pow(dt,4)*noise_ax_/4., 0, pow(dt,3)*noise_ax_/2., 0,
             0, pow(dt,4)*noise_ay_/4., 0, pow(dt,3)*noise_ay_/2.,
             pow(dt,3)*noise_ax_/2., 0, pow(dt,2)*noise_ax_, 0,
             0, pow(dt,3)*noise_ay_/2., 0, pow(dt,2)*noise_ay_;
  ekf_.Predict();

  /**
   * Update
   */

  /**
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    ekf_.R_ = R_radar_;
    Hj_ = Tools().CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
