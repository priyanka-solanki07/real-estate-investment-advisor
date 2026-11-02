# EDA insights (20 questions)

Charts are saved under `reports/figures/`. Prices in this file are synthetic and similar across cities; insights describe the table, not the Indian housing market.

## 1. Price distribution
List prices span ~10–500 lakhs and are close to uniform, which is typical of synthetic data rather than a real market with a right-skewed luxury tail.

## 2. Size distribution
Sizes fill 500–5000 sq ft almost evenly. There is no clustering around common builder-floor plates, so size is not a strong city-specific signal here.

## 3. Price per sqft by property type
Median ₹/sqft is similar across types ({'Apartment': 9245.0, 'Independent House': 9305.0, 'Villa': 9190.0}). Type does not separate unit value in this file the way it would in a real listing market.

## 4. Size vs price
Pearson correlation between size and list price is -0.003. Larger homes are not systematically more expensive, so ₹/sqft varies widely.

## 5. Outliers in INR/sqft and size
IQR flags 19,723 ₹/sqft outliers (7.9%) and 0 size outliers. Size stays inside 500–5000; high ₹/sqft is small area + high price. None were dropped.

## 6. Average INR/sqft by state
State averages sit in a narrow band (12931–13252 ₹/sqft). Do not treat this as a real interstate price ranking; the source looks synthetic.

## 7. Average price by city
City mean prices only range 250.8–258.5 lakhs. Location barely moves list price in this dataset.

## 8. Median age by locality (top 15)
Among the 15 largest localities, median age runs 18–20 years. Stock is mixed-age rather than a single construction boom.

## 9. BHK mix across cities
BHK 1–5 are spread almost evenly in every large city. Bedroom mix is not a city differentiator here.

## 10. Price trend for top 5 expensive localities (Year_Built)
There is no listing date in the file, so Year_Built is the time axis. The five priciest localities do not show a clean vintage-price trend.

## 11. Numeric correlation heatmap
The strongest structural link is list price vs ₹/sqft (r=0.56) because ₹/sqft is computed from price. Infrastructure and amenities barely correlate with price.

## 12. Schools vs INR/sqft
Average ₹/sqft is essentially flat across school counts. Nearby_Schools is not priced in this synthetic table.

## 13. Hospitals vs INR/sqft
Hospital count also shows no material ₹/sqft gradient. Do not infer a healthcare-premium from this file.

## 14. Price by furnished status
Median prices by furnishing: {'Furnished': 253.6, 'Semi-furnished': 253.8, 'Unfurnished': 254.2}. The three boxes overlap heavily; furnishing is not a price driver here.

## 15. INR/sqft by facing
North/South/East/West have indistinguishable ₹/sqft distributions. Facing has no pricing role in this data.

## 16. Count by owner type
Owner/Builder/Broker counts are nearly equal ({'Broker': 83479, 'Owner': 83268, 'Builder': 83253}). Supply is balanced by construction, not organic.

## 17. Count by availability
Ready_to_Move vs Under_Construction is a near 50/50 split ({'Under_Construction': 125035, 'Ready_to_Move': 124965}).

## 18. Parking vs price
Median price with vs without parking: {'No': 253.4, 'Yes': 254.4}. Parking does not lift list price in this table.

## 19. Amenities vs INR/sqft
Mean ₹/sqft is flat across amenity counts. Amenities were generated independently of price.

## 20. Transport vs INR/sqft and good-investment rate
₹/sqft is similar across transport levels, but the good-investment rate rises with High transport ({'Low': 0.447, 'Medium': 0.481, 'High': 0.828}) because High transport is one of the six labelling rules.
