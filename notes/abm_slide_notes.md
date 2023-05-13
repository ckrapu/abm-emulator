

**Slide 1:**

These slides show my thought process for using statistical emulators with agent based models. 

I'll step through some of the procedures to getting an emulator to work with ABM data, and try to make some of the emulator abstractions more concrete. The most common statistical model used for computer model emulation is the Gaussian process, or some variation therein. If I ever say "GP", that's what I'm referring to.

**Slide 2**

To develop these animations and results, I used the Mesa library in Python, which is very similar to NetLogo and is designed for rapid ABM experimentation.

The model I've been working with is a grid-based  SIR model in which agents can be susceptible, infected, or removed. The agents follow a random walk on the spatial graph and therefore the spread of the virus is strongly influenced by diffusion processes The animations shown here depict the spatial and temporal extent of the virus spread for two different parameter settings.

There are a couple of basic parameters which are mostly self explanatory. As shown here, there are free parameters for population size, grid properties, and infection properties. The population and grid properties will be treated as fixed. We'll use the remaining quantities as our calibration parameters of interest for the emulator.



**Slide 3**

I ran the SIR model 10 times with parameter settings randomly drawn from a uniform distribution over 4 dimensions. I also recorded the timestep, for each simulation, on which the peak infection occurred. That's called the "worst day" variable in this table. We could compute other summary statistics to model, but this one happened to be easy to interpret and calculate.

It's important to note that once we have run our ABM, the emulator will only interact with it via this table here - it knows nothing about the ABM save for the values of the input variables used to obtain the simulated response values, as well as those responses. The promise of using an emulator is that if we have a set of points which very sparsely covers the input space, we can obtain interpolated surfaces with the correct credible intervals which 1) appropriate reflect all sources of uncertainty properly, and 2) won't give silly answers and blow up at the boundaries. This latter issue can be a major problem with other interpolation methods.

