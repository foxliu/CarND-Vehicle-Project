/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = NUM_PARTICLES;
  particles.reserve(static_cast<unsigned long>(num_particles));

  random_device rd;
  default_random_engine engine(rd());
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p{};
    p.id = i;
    p.x = dist_x(engine);
    p.y = dist_y(engine);
    p.theta = dist_theta(engine);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  random_device rd;
  default_random_engine engine(rd());

  double delta_theta = yaw_rate * delta_t;

  if (fabs(yaw_rate) < LEAST_YAW_RATE) {
    for (auto &p : particles) {
      p.x += velocity * cos(p.theta) * delta_t;
      p.y += velocity * sin(p.theta) * delta_t;
      p.theta += delta_theta;

      p.x += dist_x(engine);
      p.y += dist_y(engine);
      p.theta += dist_theta(engine);
    }
  } else {
    double r = velocity / yaw_rate;
    for (auto &p : particles) {
      p.x += r * (sin(delta_theta + p.theta) - sin(p.theta));
      p.y += r * (cos(p.theta) - cos(delta_theta + p.theta));
      p.theta += delta_theta;

      p.x += dist_x(engine);
      p.y += dist_y(engine);
      p.theta += dist_theta(engine);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto &observation : observations) {
    double min_dist = numeric_limits<double>::max();
    int m_id = -1;
    for (const auto &pd : predicted) {
      double tmp_dist = dist(observation.x, observation.y, pd.x, pd.y);
      if (tmp_dist < min_dist) {
        min_dist = tmp_dist;
        m_id = pd.id;
      }
    }
    observation.id = m_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  const double sig_x = std_landmark[0];
  const double sig_y = std_landmark[1];
  const double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
  const double c_x = 2.0 * sig_x * sig_x;
  const double c_y = 2.0 * sig_y * sig_y;

  weights.clear();

  for (auto &p : particles) {
    // get the landmarks where in sensor range
    vector<LandmarkObs> sensor_landmarks;
    for (auto &landmark : map_landmarks.landmark_list) {
      if (dist(landmark.x_f, landmark.y_f, p.x, p.y) < sensor_range) {
        LandmarkObs obs{};
        obs.id = landmark.id_i;
        obs.x = landmark.x_f;
        obs.y = landmark.y_f;
        sensor_landmarks.push_back(obs);
      }
    }
    if (sensor_landmarks.size() == 0) {
      sensor_landmarks.push_back(LandmarkObs{p.id, p.x, p.y});
    }

    // coordinate transform
    vector<LandmarkObs> transform_ob_to_map;
    transform_ob_to_map.reserve(observations.size());

    const double cos_p = cos(p.theta);
    const double sin_p = sin(p.theta);

    for (const auto &observation : observations) {
      double map_x = observation.x * cos_p - observation.y * sin_p + p.x;
      double map_y = observation.x * sin_p + observation.y * cos_p + p.y;
      LandmarkObs ttm{observation.id, map_x, map_y};
//      ttm.id = observation.id;
//      ttm.x = map_x;
//      ttm.y = map_y;
      transform_ob_to_map.push_back(ttm);
    }

//    for (const auto &tm : transform_ob_to_map) {
//      cout << "+++++++++++++++++++=" << endl;
//      cout << "tm.x: " << tm.x << "\ttm.y: " << tm.y << "\ttm.id: " << tm.id << endl;
//    }

    dataAssociation(sensor_landmarks, transform_ob_to_map);
    // transform_ob_to_map id is the nearest sensor_landmarks's id

    // update weights
    double weight = 1.0;

//    cout << "======================" << endl;
    for (auto &t_m : transform_ob_to_map) {
      for (auto &sl : sensor_landmarks){
        if (sl.id == t_m.id) {
//          cout << "sl.id: " << sl.id << "\tt_m.id: " << t_m.id << "\tsl.x: " << sl.x << "\tsl.y: " << sl.y;
//          cout << "\tt_m.x: " << t_m.x << "\tt_m.y: " << t_m.y << endl;
          weight *= gauss_norm * exp(-(pow(t_m.x - sl.x, 2) / c_x + (pow(t_m.y - sl.y, 2) / c_y)));
          break;
        }
      }
    }
    p.weight = weight;
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

//  vector<Particle> new_particles;
//
//  default_random_engine engine;
//
//  uniform_int_distribution<int> random_index(0, num_particles-1);
//  int index = random_index(engine);
//  double beta = 0;
//  double mw = *max_element(weights.begin(), weights.end());
//  uniform_real_distribution<double> random_beta(0.0, 1.0);
//  for (int j = 0; j < num_particles; ++j) {
//    beta += random_beta(engine) * 2 * mw;
//    while (beta > weights[index]) {
//      beta -= weights[index];
//      index = (index + 1) % num_particles;
//    }
//    new_particles.push_back(particles[index]);
//  }
//  particles.clear();
//  particles.insert(particles.end(), new_particles.begin(), new_particles.end());
  random_device seed;
  mt19937 random_generator(seed());
  discrete_distribution<> sample(weights.begin(), weights.end());

  vector<Particle> new_particles(num_particles);
  for (auto &p : new_particles) {
    p = particles[sample(random_generator)];
  }
  particles = move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
