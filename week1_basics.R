# Load the tidyverse
library(tidyverse)

# Load a built-in dataset
data <- mtcars

# Convert rownames to a column
data <- data %>%
  rownames_to_column(var = "car")

# Look at the first few rows
head(data)

# Filter: cars with mpg > 20
filtered_data <- data %>%
  filter(mpg > 20)

# Summarize: average mpg by cylinder count
summary <- data %>%
  group_by(cyl) %>%
  summarize(avg_mpg = mean(mpg), .groups = "drop")

print(summary)

# Plot: mpg vs hp
ggplot(data, aes(x = hp, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  labs(
    title = "Miles per Gallon vs. Horsepower",
    x = "Horsepower",
    y = "Miles per Gallon",
    color = "Cylinders"
  ) +
  theme_minimal()

