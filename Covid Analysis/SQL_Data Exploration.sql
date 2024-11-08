-- Project 

/* Covid 19 Data Exploration

Skills applied: Joins, CTE'S , Temp Tables, Window Functions, Aggregate Functions, Creating Views, Converting Data Types */


-- Our Tables in database
SELECT 
    *
FROM
    dataexploration.CovidVaccination
ORDER BY 3 , 4;


SELECT 
    *
FROM
    dataexploration.CovidDeath
ORDER BY 3 , 4;




-- Select data that we are using for the project 

SELECT 
    location,
    date,
    total_cases,
    new_cases,
    total_deaths,
    (total_deaths / total_cases) * 100 AS DeathPercentage
FROM
    dataexploration.CovidDeath
WHERE
    location LIKE '%Nepal%'
ORDER BY 1 , 2;




-- Total case vs Total Deaths
-- Shows what percentage of population is likely to die if they get Covid in Nepal

SELECT 
    location,
    date,
    total_cases,
    Population,
    (total_deaths / population) * 100 AS DeathPercentage
FROM
    dataexploration.CovidDeath
WHERE
    location LIKE '%Nepal%'
    AND 
    continent is not null 
ORDER BY 1 , 2;



-- Total Cases vs Population 
-- Shows what percentage of population get Covid in Nepal
SELECT 
    location,
    date,
    total_cases,
    Population,
    (total_deaths / population) * 100 AS DeathPercentage
FROM
    dataexploration.CovidDeath
/* WHERE
    location LIKE '%Nepal%'
    AND 
    continent is not null  */
ORDER BY 1 , 2;



/* WHERE
    location LIKE '%Nepal%' */ 
-- Looking at Countries with Highest Infection Rate comapred to Population
SELECT 
    Location,
    population,
    MAX(total_cases) AS HighestInfectionCount,
    MAX((total_cases / population)) * 100 AS PercentPopulationInfected
FROM
    dataexploration.CovidDeath
GROUP BY Location , Population
ORDER BY PercentPopulationInfected DESC;




-- Showing Countries with the highest Death count per population

Select Location,  MAX(cast(total_deaths as float)) as TotalDeathCount
From dataexploration.CovidDeath
where continent is not null 
Group by Location
order by TotalDeathCount desc;




-- BREAK DOWN BY CONTINENT
-- Showing continents with highest death counts per population

Select continent,  MAX(cast(total_deaths as float)) as TotalDeathCount
From dataexploration.CovidDeath
where continent is not null 
Group by continent
order by TotalDeathCount desc;



-- GLOBAL NUMBERS:
	-- Using Aggregate Functions
    
Select SUM(new_cases), SUM(cast(new_deaths as float)), SUM(new_deaths) /SUM(cast(new_deaths as float))/ SUM(new_cases) *100 as DeathPercentage
FROM dataexploration.CovidDeath
where continent is not null
order by 1, 2; 



-- Covid Vaccinations and Covid Death Table JOIN

SELECT 
    *
FROM
    dataexploration.CovidDeath dea
        JOIN
    dataexploration.CovidVaccination vac ON dea.location = vac.location
        AND dea.date = vac.date;




-- Total Population vs Vaccination

SELECT 
    dea.continent,
    dea.location,
    dea.date,
    dea.population,
    vac.new_vaccinations
FROM
    dataexploration.CovidDeath dea
        JOIN
    dataexploration.CovidVaccination vac ON dea.location = vac.location
        AND dea.date = vac.date
WHERE
    dea.continent IS NOT NULL
ORDER BY 2 , 3;




-- Shows Percentage of Population that has received at least one Covid Vaccine 

Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as float)) OVER(Partition by dea.location Order by dea.location, dea.Date) as RollingPeopleVaccinated
From dataexploration.CovidDeath dea
JOIN dataexploration.CovidVaccination vac
On dea.location = vac.location
and dea.date = vac.date
where dea.continent is not null
order by 2, 3;



-- USE of CTE 
	-- Performed Calculation on Partition By in previous query
    
With PopvsVac (Continent, Location, Date, Population,New_Vaccinations, RollingPeopleVaccinated)
as (
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as float)) OVER(Partition by dea.location Order by dea.location, dea.Date) as RollingPeopleVaccinated
-- (RollingPeopleVaccinated/Population)*100 
From dataexploration.CovidDeath dea
JOIN dataexploration.CovidVaccination vac
On dea.location = vac.location
and dea.date = vac.date
where dea.continent is not null
-- order by 2, 3
)
Select *, (RollingPeopleVaccinated / Population)* 100
From PopvsVac;



-- TEMP TABLE
	-- Performed Calculation on Partition By in previous query
USE dataexploration;
DROP table if exists PercentPopulationVaccinated ;
Create Table PercentPopulationVaccinated
( 
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric, 
RollingPeopleVaccinated numeric 
) ;
Insert into PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as float)) OVER(Partition by dea.location Order by dea.location, dea.Date) as RollingPeopleVaccinated
From dataexploration.CovidDeath dea
JOIN dataexploration.CovidVaccination vac
On dea.location = vac.location
and dea.date = vac.date ;
-- where dea.continent is not null;
-- order by 2, 3; 

Select *, (RollingPeopleVaccinated/ Population) * 100
From PercentPopulationVaccinated;




-- Create View 
-- Use dataexploration database;
	-- For Data Visualization 
Create View PercentPoplnVaccinated as 
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(cast(vac.new_vaccinations as float)) OVER(Partition by dea.location Order by dea.location, dea.Date) as RollingPeopleVaccinated
From dataexploration.CovidDeath dea
JOIN dataexploration.CovidVaccination vac
On dea.location = vac.location
and dea.date = vac.date 
where dea.continent is not null;
-- order by 2, 3; 


-- Work table for Tableau
Select * From Percentpoplnvaccinated;



