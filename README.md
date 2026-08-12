# Replicating VIX (US Market Volatility Index) for Brazil


## Objective
In this project, I worked on creating a VIX equivalent for the Brazilian market, starting with no knowledge of B3's data structure. The process was full of challenges, like dealing with messy and incomplete data, which didn't follow the clean examples I found on reference textbooks. But instead of getting stuck on these issues, I focused on finding ways to work around them.

## Description
One of the biggest hurdles was adapting the VIX methodology to Brazil's market, where data is harder to come by and often inconsistent. I had to get creative with how I validated my results and handled the gaps. There were no ready-made solutions or benchmarks, so I had to troubleshoot and figure things out on my own. The project prioritized methodological robustness over theoretical purity, focusing on producing usable volatility signals under real-world data constraints.

## Steps
  Obtained Options Data: I used an API to pull options chain data and stored it in a PostgreSQL database.
  Daily Data Processing: Each day, I queried the database to retrieve the latest options chains.
  Methodology Adaptation: I selected options adapting the methodology for the data I had.
  Daily VIX Calculation: I calculated and plotted the daily VIX (future volatility) levels.
  Actual Volatility Calculation: I downloaded Ibovespa prices to calculate actual volatility and compared it with my index estimates.

Pedro Todescan 

