
#![feature(drain_filter)]

use serde::{Serialize, Deserialize};
use pyo3::prelude::*;
use pyo3::types::*;
use rand::seq::SliceRandom;
use log::{info, debug};
use std::fmt;
use std::sync::Arc;
use rand::prelude::*;
use dict_derive::{FromPyObject, IntoPyObject};
use std::cmp::{min,max};
use hashbrown::{HashMap};
use simplelog::*;
use std::fs::File;
use rayon::prelude::*;


#[derive(Clone,PartialEq,Debug,Serialize,Deserialize)]
pub enum JobType {
    Pickup,
    Delivery
}

#[derive(Clone,Debug,Default,Serialize,Deserialize)]
pub struct TimeWindow {
    start: f32,
    end: f32,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Job {
    name: String,
    id : usize,
    loc: usize,
    job_type: JobType,
    tw: TimeWindow,
    weight: f32,
    service_time: f32,
    x: f32,
    y: f32,
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct DistTime {
    dist: f32,
    time: f32,
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Vehicle {
    cap: f32,
    tw: TimeWindow,
    start_loc: usize,
    end_loc: usize,
    fee_per_dist: f32,
    fee_per_time: f32,
    fixed_cost: f32,
    handling_cost_per_weight: f32,
    max_stops: usize,
    max_dist: f32,
}


#[derive(Default,Clone,Debug,Serialize,Deserialize,FromPyObject, IntoPyObject)]
pub struct State {
    dist: f32,
    time: f32,
    stops: usize,
    time_slack: f32,
    wait_time: f32,
    service_time: f32,
    loc: usize,
    weight: f32,
}



#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Tour {
    tour: Vec<usize>,
    vehicle: Vehicle,
    states: Vec<State>,
}

impl Tour {
    pub fn new(v: Vehicle,cap: usize) -> Self {
        Self {
            tour: Vec::with_capacity(cap),
            vehicle: v,
            states: Vec::with_capacity(cap)
        }
    }

    #[inline(always)]
    fn calc_delta(&self,delta_d: f32,delta_t: f32,delta_w: f32) -> f32 {
        let mut delta = delta_d * self.vehicle.fee_per_dist + delta_w * self.vehicle.handling_cost_per_weight;
        // self.vehicle.fee_per_time * delta_t
        if self.tour.len() == 0 {
            delta += self.vehicle.fixed_cost;
        }
        delta
    }

    #[inline(always)]
    fn calc_cost(&self) -> f32 {
        let last_state = self.states.last().unwrap();
        // last_state.leave_time * self.vehicle.fee_per_time
        last_state.dist * self.vehicle.fee_per_dist   + last_state.weight * self.vehicle.handling_cost_per_weight + self.vehicle.fixed_cost
    }
    
    
}



#[derive(Clone,Debug)]
pub struct VehicleManager {
    vehicles: Vec<Vehicle>,
    dist_time: Arc<Vec<Vec<DistTime>>>,
}

impl VehicleManager {
    pub fn new(vehicles: Vec<Vehicle>,dist_time: Arc<Vec<Vec<DistTime>>>) -> Self {
        Self {
            vehicles: vehicles,
            dist_time: dist_time,
        }
    }

    pub fn alloc(&self,j: &Job) -> Vehicle {
        self.vehicles.iter().min_by(|x,y|{
            self.dist_time[x.start_loc][j.loc].dist.partial_cmp(&self.dist_time[y.start_loc][j.loc].dist).unwrap()
        }).unwrap().clone()
    }
}


#[derive(Clone,Debug)]
pub struct Solution {
    vm: VehicleManager,
    jobs: Vec<Job>,
    tours: Vec<Tour>,
    absents: Vec<usize>,
    pub cost: f32,
}

impl Solution {
    fn new(jobs: Vec<Job>, vm: VehicleManager) -> Self {
        let len = jobs.len();
        Self {
            vm: vm,
            jobs: jobs,
            tours: Vec::with_capacity(100),
            absents: (0..len).collect(),
            cost: 0f32,
        }
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cost: {}, vehicles: {}, absents: {}", self.cost, self.tours.len(), self.absents.len())
    }
}


pub struct Recreate {
    dist_time: Arc<Vec<Vec<DistTime>>>,
    cost_per_absent: f32,
}


impl Recreate {

    pub fn new(dist_time: Arc<Vec<Vec<DistTime>>>,cost_per_absent: f32) -> Self {
        Self {
            dist_time: dist_time,
            cost_per_absent: cost_per_absent,
        }
    }

    #[inline(always)]
    pub fn calc_cost(&self,s: &mut Solution) {
        let mut cost: f32 = s.tours.iter().map(|x| { x.calc_cost() }).sum();
        cost += self.cost_per_absent * s.absents.len() as f32;
        s.cost = cost;
    }

    pub fn check_states(&self,s: &Solution) {
        let tours = &s.tours;
        for t in tours.iter() {
            self.check_tour_states(&*s.jobs, t);
        }
    }

    pub fn check_tour_states(&self,jobs: &[Job],t: &Tour) {
        let tour = &t.tour;
        let vehicle = &t.vehicle;

        let mut t = vehicle.tw.start;
        let mut weight = 0.0f32;
        let mut prev_loc = vehicle.start_loc;
        let mut last_service_time = 0f32;

        for x in tour.iter() {
            let j = &jobs[*x];
            let dist_time = &self.dist_time[prev_loc][j.loc];
            t += dist_time.time + last_service_time;

            if t < j.tw.start {
                t = j.tw.start;
            }

            // assert!( t <= j.tw.end );
            if t > j.tw.end {
                debug!("tour: {:?}, x: {:?}, t: {:?}, end: {:?}",tour,x,t,j.tw.end);
            }
            weight += j.weight;
            if weight > vehicle.cap {
                debug!("tour: {:?}, weight: {:?}, cap: {:?}",tour,weight,vehicle.cap);
            }
            prev_loc = j.loc;
            last_service_time = j.service_time;
        }
        
        let dist_time = &self.dist_time[prev_loc][vehicle.end_loc];
        t += dist_time.time + last_service_time;
        assert!( t <= vehicle.tw.end );

    }

    #[inline(always)]
    pub fn create_tour_states(&self,jobs: &[Job],t: &mut Tour) {
        let tour = &t.tour;
        let vehicle = &t.vehicle;
        let states = &mut t.states;
        
        //vehicle start loc..
        if states.len() == 0 {
            let first_state = State {
                time: vehicle.tw.start,
                time_slack: vehicle.tw.end - vehicle.tw.start,
                loc: vehicle.start_loc,
                ..State::default()
            };
            states.push(first_state);
        }

        //for all jobs
        for x in tour.iter().skip(states.len()-1) {
            let state = &states[states.len()-1];
            let j = &jobs[*x];
            let mut new_state = state.clone();
            let dist_time = &self.dist_time[state.loc][j.loc];

            new_state.dist += dist_time.dist;
            new_state.time += state.service_time + dist_time.time;

            if new_state.time < j.tw.start {
                new_state.wait_time = j.tw.start - new_state.time;
                new_state.time = j.tw.start;
            } else {
                new_state.wait_time = 0f32;
            }

            new_state.time_slack = j.tw.end - new_state.time;
           
            new_state.stops += if state.loc == j.loc {
                0
            } else {
                1
            };

            new_state.weight += j.weight;
            new_state.loc = j.loc;
            new_state.service_time = j.service_time;

            states.push(new_state);
        }


        //vehicle end loc..
        let state = &states[states.len()-1];
        let mut last_state = state.clone();
        let dist_time = &self.dist_time[state.loc][vehicle.end_loc];
        last_state.dist +=  dist_time.dist;
        last_state.time = state.service_time + dist_time.time;
        last_state.wait_time = 0f32;
        last_state.service_time = last_state.service_time;
        last_state.time_slack = vehicle.tw.end - last_state.time;
        last_state.loc = vehicle.end_loc;

        let mut last_time_slack = last_state.time_slack;
        let mut last_wait_time = last_state.wait_time;
        states.push(last_state);

        for state in states.iter_mut().rev().skip(1) {
            state.time_slack = state.time_slack.min(last_time_slack+last_wait_time);
            last_time_slack = state.time_slack;
            last_wait_time = state.wait_time;
        }

    }


    fn create_states(&self,s: &mut Solution) {
        let tours = &mut s.tours;
        for t in tours.iter_mut() {
            if t.tour.len()+2 > t.states.len() {
                self.create_tour_states(&*s.jobs, t)
            }
        }
    }

    #[inline(always)]
    fn try_insert_tour(&self,tour: &Tour,job: &Job) -> (usize,f32) {
        let states = &tour.states;
        let v = &tour.vehicle;
        let last_state = &states[states.len()-1];
        let mut pos = 0 as usize;
        let mut best = std::f32::MAX;


        if job.weight + last_state.weight > v.cap {
            return (0,std::f32::MAX);
        }

        for (j,w) in states.windows(2).enumerate() {

            let delta_stops = if w[0].loc != job.loc && w[1].loc != job.loc {
                1
            } else {
                0
            };

            //check max_stops
            if v.max_stops > 0 && delta_stops + last_state.stops > v.max_stops {
                continue;
            }

            let dt1 = &self.dist_time[w[0].loc][job.loc];
            let dt2 = &self.dist_time[job.loc][w[1].loc];
            let dt3 = &self.dist_time[w[0].loc][w[1].loc];

            //check max_dist
            let delta_d =  dt1.dist + dt2.dist - dt3.dist;
            if v.max_dist > 0f32 && delta_d + last_state.dist > v.max_dist {
                continue;
            }

            

            let mut t = w[0].time;
            t += dt1.time + w[0].service_time;
            //check self time window
            if t > job.tw.end {
                break;
            }

            if t < job.tw.start {
                t = job.tw.start
            }

            t += job.service_time + dt2.time;

            debug!("===========>job:{:?},t:{:?},w[1]:{:?}",job,t,w[1]);

            //check time slack
            // let delta_t = t - w[1].arrive_time;// to be fixed
            let delta_t = 0f32;
            if t - w[1].time> w[1].time_slack {
                debug!("===========>check faild");
                continue;
            }

            let delta_cost = tour.calc_delta(delta_d,delta_t,job.weight);
            debug!("===========>delta: {:?},delta_d: {:?},delta_t: {:?}",delta_cost,delta_d,delta_t);
            
            if delta_cost < best {
                pos = j;
                best = delta_cost;
            }

        }

        (pos,best)
    }


    fn try_insert(&self,tours: &[Tour],job: &Job) -> (usize,usize,f32) {
        let mut pos = (0,0);
        let mut best = std::f32::MAX;

        for (i,tour) in tours.iter().enumerate() {
            let (j,delta) = self.try_insert_tour(tour, job);
            if delta < best {
                best = delta;
                pos = (i,j);
            }
        }

        (pos.0,pos.1,best)

    }


    #[inline(always)]
    fn do_insert(&self,jobs: &[Job],tour: &mut Tour,j: usize,x: usize) {
        debug!("tour: {:?}",tour.tour);
        tour.tour.insert(j, x);
        tour.states.drain(j..);
        self.create_tour_states(jobs,tour);
    }

    fn ruin(&self,s: &mut Solution,jobs: Vec<usize>) {
        let tours = &mut s.tours;

        tours.drain_filter(| tour | {
            let removed = tour.tour.drain_filter(|x| {
                jobs.contains(x)
            }).count();

            if removed > 0 {
                tour.states.clear();
            }

            tour.tour.len() == 0
            
        });
        s.absents = jobs;
        // absents.push(tour.tour.remove(index));  
        // tour.states.drain(index+1..);
    }

    fn recreate(&self, s: &mut Solution,random: bool) {
        // info!("before recreate tour lens:{:?},{:?}", s.tours.iter().map(|x| x.tour.len()).sum::<usize>(),s.absents.len());

        self.create_states(s);
        let absents = &mut s.absents;
        let tours = &mut s.tours;
        let jobs = &s.jobs;
        let vm = &s.vm;

        if random {
            let mut rng = rand::thread_rng();
            absents.shuffle(&mut rng);
        }

        absents.drain_filter(|x| {
            let job = &jobs[*x];
            let (t,j,delta) = self.try_insert(tours,job);
            if delta == std::f32::MAX {
                // info!("try add new ve: {}, {}",tours.len(),s.max_tours);
                let v = vm.alloc(job);
                let mut tour = Tour::new(v,100);
                self.create_tour_states(jobs, &mut tour);
                let (j,delta) = self.try_insert_tour(&tour, job);
                if delta == std::f32::MAX {
                    // info!("failed to add new ve: {}, {}",tours.len(),s.max_tours);
                    // info!("failed to add new tour ====>  tour.states:{:?},job:{:?}",tour.states,job);
                    return false;
                }
                self.do_insert(jobs, &mut tour, j, *x);
                tours.push(tour);
              
            } else {
                let tour = &mut tours[t];
                self.do_insert(jobs, tour, j, *x);
            }
            true

        });

        self.calc_cost(s);
        // info!("after recreate tour lens:{:?},{:?}", s.tours.iter().map(|x| x.tour.len()).sum::<usize>(),s.absents.len());

    }
}



#[derive(Debug,Serialize,Deserialize)]
pub struct Input {
    vehicles: Vec<Vehicle>,
    dist_time: Arc<Vec<Vec<DistTime>>>,
    cost_per_absent: f32,
    jobs: Vec<Job>,
    temperature: f32,
    c2: f32,
    sa: bool,
}


pub struct EnvInner {
    sol: Solution,
    re: Recreate,
    temperature: f32,
    c: f32,
    sa: bool,
}

impl EnvInner {

    fn new(input_str: &str) -> EnvInner {
        let input: Input = serde_json::from_str(&input_str).unwrap();
        let vm = VehicleManager::new(input.vehicles, input.dist_time.clone());
        let mut sol = Solution::new(input.jobs, vm);

        let re = Recreate::new(input.dist_time.clone(), input.cost_per_absent);
        re.recreate(&mut sol, false);

        EnvInner {
            sol: sol,
            re: re,
            temperature: input.temperature,
            c: input.c2,
            sa: input.sa,
        }
    }


    fn states(&self) -> Vec<Vec<State>> {
        let states: Vec<Vec<State>> = self.sol.tours.iter().map(|x| x.states.clone() ).collect();
        states
    }

    fn tours(&self) -> Vec<Vec<usize>> {
        let tours: Vec<Vec<usize>> = self.sol.tours.iter().map(|x| x.tour.clone() ).collect();
        tours
    }

    
    fn step(&mut self,jobs: Vec<usize>)  {
        if self.sa {
            let mut cur = self.sol.clone();
            let s = &mut cur;
            self.re.ruin(s, jobs);
            self.re.recreate(s,false);

            let mut rng = rand::thread_rng();
            let new_cost = s.cost;
            let old_cost = self.sol.cost;
            if new_cost < old_cost || new_cost < old_cost - self.temperature * rng.gen::<f32>().ln() {
                self.sol = cur;
            }
            self.temperature *= self.c;

        } else {
            let s = &mut self.sol;
            self.re.ruin(s, jobs);
            self.re.recreate(s,false);
        }

    }

    fn absents(&self) -> Vec<usize> {
        self.sol.absents.clone()
    }

    fn cost(&self) -> f32 {
        self.sol.cost
    }
}



#[pyclass]
pub struct Env {
    inner: EnvInner
}


#[pymethods]
impl Env {

    #[new]
    fn __new__(obj: &PyRawObject,py: Python,input_str: &str) -> PyResult<()> {
        let inner = EnvInner::new(input_str);
        obj.init(Env {
            inner: inner,
        });

        Ok(())
    }


    fn states(&self, py: Python) -> PyResult<Vec<Vec<State>>> {
        Ok(self.inner.states())
    }

    fn tours(&self, py: Python) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.inner.tours())
    }

    fn step(&mut self,py: Python,jobs: Vec<usize>) -> PyResult<()> {
        Ok((self.inner.step(jobs)))
    }

    fn absents(&self,py: Python) -> PyResult<Vec<usize>> {
        Ok(self.inner.absents())
    }

    fn cost(&self,py: Python) -> PyResult<f32> {
        Ok(self.inner.cost())
    }

}


#[pyclass]
pub struct BatchEnv {
    envs: Vec<EnvInner>,
}


#[pymethods]
impl BatchEnv {

    #[new]
    fn __new__(obj: &PyRawObject,py: Python,input_strs: Vec<String>) -> PyResult<()> {
        let envs: Vec<EnvInner> = input_strs.iter().map(|x| {
            EnvInner::new(x)
        }).collect();
        obj.init(BatchEnv {
            envs: envs,
        });
        Ok(())
    }


    fn states(&self, py: Python) -> PyResult<Vec<Vec<Vec<State>>>> {
        let ret = self.envs.iter().map(|x|{
            x.states()
        }).collect::<Vec<_>>();
        Ok(ret)
    }

    fn tours(&self, py: Python) -> PyResult<Vec<Vec<Vec<usize>>>> {
        let ret = self.envs.iter().map(|x|{
            x.tours()
        }).collect::<Vec<_>>();
        Ok(ret)
    }

    
    fn step(&mut self,py: Python,jobs: Vec<Vec<usize>>) -> PyResult<()> {
        self.envs.iter_mut().zip(jobs.iter().cloned()).for_each(|x|{
            x.0.step(x.1)
        });
        Ok(())
    }

    fn absents(&self,py: Python) -> PyResult<Vec<Vec<usize>>> {
        let ret = self.envs.iter().map(|x|{
            x.absents()
        }).collect::<Vec<_>>();
        Ok(ret)
    }

    fn cost(&self,py: Python) -> PyResult<Vec<f32>> {
        let ret = self.envs.iter().map(|x|{
            x.cost()
        }).collect::<Vec<_>>();
        Ok(ret)
    }
}


#[pymodule]
fn vrp_env(py: Python, m: &PyModule) -> PyResult<()> {
    CombinedLogger::init(
        vec![
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create("log.log").unwrap()),
        ]
    ).unwrap();
    m.add_class::<Env>()?;
    m.add_class::<BatchEnv>()?;
    Ok(())
}





