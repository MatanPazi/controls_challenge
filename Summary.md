
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

## Defining the state vector, $x_k$:

According to the costs in this challenge, the current and previous lateral acceleration are sufficient to calculate both, so those 2 states are crucial to have in the state vector.

I need to include the previous steering commands because, due to the delay, they will affect the resulting lateral acceleration and lateral jerk.
I'm also adding the accelerometer bias, since using the observer I'll be able to estimate it and use it to get more accurate sensor laterala cceleration measurments.

- $ay_k$  
- $ay_{k-1}$  
- $delta_{k-1}$  
- $delta_{k-2}$  
- $delta_{k-3}$ 
- $b_k$

### Legend
- ay — lateral acceleration  
- delta — steering command  
- b — IMU lateral-accel bias (random walk)

## Exogenous Inputs (ζₖ)

Next are the exogenous inputs. I'm not interested in them directly, nor do I command them, but they affect the system response so they're taken into account.

- $vx_k$  
- $ax_k$  
- $r_k$

### Legend
- $vx_k$ — longitudinal speed  
- $ax_k$ — longitudinal acceleration  
- $r_k$  — road roll lateral acceleration

## State Update

$$
\mathbf{x}_{k+1} = A(\zeta_k) \mathbf{x}_k + B \mathbf{u}_k + E(\zeta_k) \zeta_k + \mathbf{w}_k
$$

## System identification

### Delay

The effective sample delay in the TinyPhysics ONNX model was identified.
This was done by comparing lateral acceleration trajectories between a zero-steer controller and a constant step-steer controller on real driving segments.

The delay was measured by how many simulation steps pass after control start (CONTROL_START_IDX) 
before the step response noticeably diverges from the zero-steer baseline.

The result was varried. A minimum of 1 sample delay and a maximum of 5 samples were detected.
Though the majority of routes exhibited 2 sample delays, with a linearly decreasing of routes exhibiting higher sample delays.

As an initial step, I'll include the last three previous steering commands.
I'd rather keep the state dimension as small as possible.
The # of sample delays can be adjusted later if needed.

### Noise

To be determined later, Though, from a quick inspection, a skewed gaussian was detected.

How to determine exactly?

This will be used to determine which observer to use.

## Planned controller

Stochastic MPC



## LLM, ignore this: 
### TODO:
* Identify model assuming a LPV ARX model. Assume only dependence on vEgo for the moment.

* Designing a controller:
  * Stochastic LPV MPC

* Noise identification (Process + measurement)
