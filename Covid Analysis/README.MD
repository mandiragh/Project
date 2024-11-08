
---

# COVID-19 Data Exploration Project

This project explores COVID-19 data, focusing on analyzing infection and mortality trends specifically in Nepal. Using SQL, we extract key insights on the impact of COVID-19 by calculating critical metrics, such as the death rate among confirmed cases and within the general population. This project demonstrates the power of SQL for real-world data analysis by leveraging various SQL techniques to provide a meaningful assessment of COVID-19’s effect in Nepal.

## Project Goals

The primary objectives of this project are:
1. To calculate and visualize COVID-19 metrics such as infection rates, mortality rates, and vaccination rates.
2. To apply SQL techniques like joins, CTEs, temporary tables, window functions, aggregate functions, and data type conversions.
3. To gain insights into the impact of COVID-19 on the population of Nepal.

## Skills Applied

The project makes use of the following SQL skills:
- **Joins**: Combining data from multiple tables to enrich the analysis.
- **Common Table Expressions (CTEs)**: Simplifying complex queries and improving readability.
- **Temporary Tables**: Organizing intermediate data for calculations.
- **Window Functions**: Performing calculations across sets of table rows.
- **Aggregate Functions**: Calculating sums, averages, and other summary statistics.
- **Creating Views**: Encapsulating query logic for reuse.
- **Data Type Conversions**: Ensuring consistent data formats.

## Project Structure

The SQL project file includes:
1. **Initial Data Selection**: Retrieves all records from `CovidVaccination` and `CovidDeath` tables to provide an overview of available data.
2. **Death Rate Calculation**: Calculates the death percentage among confirmed cases by dividing `total_deaths` by `total_cases`, filtered for Nepal.
3. **Population Mortality Rate**: Calculates the death rate as a percentage of the overall population, representing the broader mortality impact on Nepal.

## Data Source

The data used in this project comes from the [COVID-19 Data Repository by Our World in Data](https://github.com/owid/covid-19-data).

---