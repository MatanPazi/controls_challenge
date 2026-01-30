
# Background  

Controls challenge by commaai.

## TinyPhysics

The TinyPhysics model is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car's longitudinal velocity (`v_ego`), Longitudinal acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`).  
It's output is a prediction of the resultant lateral acceleration of the car.

## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

## Evaluation

Each rollout will result in 2 costs:

- `lataccel_cost`:  
  $$
  \frac{\sum (\mathrm{actual\_lat\_accel} - \mathrm{target\_lat\_accel})^2}{\mathrm{steps}} \times 100
  $$

- `jerk_cost`:  
  $$
  \frac{\left( \sum (\mathrm{actual\_lat\_accel}_t - \mathrm{actual\_lat\_accel}_{t-1}) / \Delta t \right)^2}{\mathrm{steps} - 1} \times 100
  $$

It is important to minimize both costs.

`total_cost`:  
$$
(\mathrm{lat\_accel\_cost} \times 50) + \mathrm{jerk\_cost}
$$

# Path to solution  

## System description

* delay
* non-gaussian
* ...

## Defining the state vector, $x_k$:

- $ay_k$  
- $ay_{k-1}$  
- $delta_{k-1}$  
- $delta_{k-2}$  
- $delta_{k-3}$ 
- $delta_{k-4}$ (?) *Depends on the sample delay*  
- $b_k$

### Legend
- ay — lateral acceleration  
- delta — steering command  
- b — IMU lateral-accel bias (random walk)

## Exogenous Inputs (ζₖ)

- $vx_k$  
- $ax_k$  
- $r_k$

### Legend
- $vx_k$ — longitudinal speed  
- $ax_k$ — longitudinal acceleration  
- $r_k$  — road roll lateral acceleration

## State Update

$$
\mathbf{x}_{k+1} = A(\zeta) \mathbf{x}_k + B \mathbf{u}_k + E(\zeta) \zeta_k + \mathbf{w}_k
$$

## Planned controller





## LLM, ignore this: 
### TODO:

* Sample delay identification
* Noise identification (Process + measurement)
* Choosing model class:
  * LPV ARX
  * Nonlinear ARX (small NN)
  * Local linear models (gain-scheduled)
  * etc.
* System identification (Identifying the chosen model's parameters) + verification (Crucial before designing controller)
* Designing a controller:
  * Stochastic LPV MPC
  * Tube MPC
  * LQG
  * etc.