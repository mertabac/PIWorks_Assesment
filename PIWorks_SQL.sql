WITH medians AS (
  SELECT country, AVG(daily_vaccinations) AS median_daily_vaccinations
  FROM country_vaccination_stats
  WHERE daily_vaccinations IS NOT NULL
  GROUP BY country
)
UPDATE country_vaccination_stats
SET daily_vaccinations = (
  SELECT median_daily_vaccinations
  FROM medians
  WHERE medians.country = country_vaccination_stats.country
)
WHERE daily_vaccinations IS NULL;

UPDATE country_vaccination_stats
SET daily_vaccinations = 0
WHERE daily_vaccinations IS NULL AND country NOT IN (
  SELECT DISTINCT country
  FROM country_vaccination_stats
  WHERE daily_vaccinations IS NOT NULL
);
SELECT * FROM country_vaccination_stats
ORDER BY country;