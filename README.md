G-PRO is a Gaussian Process Regression learning model that takes in the inputs of center of pressure, factor of safety, and weight to determine the tip chord, root chord, and semi-span of the fins

The goal is to augment the original code to take the trained data, find the best range of sustainer fin sizes and use that combined with the booster fins to decide size
Or
Is it better to have a picked sustainer fin then a booster fin to accompany it

Questions needed answered:
How to go about choosing the fins
Factor of safety goal, center of pressure goal?

Thickness and density of new fins
New bounds? (Span min of 4.8 in)

Need to adjust G-PRO to take sweptback fins for the sustainer

Need to adjust to take two sets of CP min and max

Currently takes 10 inputs
The tip and root chord, semi-span, sweep angle, and cp of the sustainer
The tip and root chord, semi-span of the booster
Total rocket Fos and cp
