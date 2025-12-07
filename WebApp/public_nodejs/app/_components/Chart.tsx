"use client";

import { useMemo } from "react";
import {
	Area,
	AreaChart,
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import type { DataLegend } from "../page";

// // Sample data for Polish companies
// const generateData = (start: string, end: string) => {
// 	const startDate = new Date(start);
// 	const endDate = new Date(end);
// 	const data = [];

// 	const current = new Date(startDate);
// 	while (current <= endDate) {
// 		const year = current.getFullYear();
// 		const quarter = Math.floor(current.getMonth() / 3) + 1;
// 		const dateStr = `${year} Q${quarter}`;

// 		data.push({
// 			date: dateStr,
// 			revenue: 100 + Math.random() * 50 + (year - 2020) * 15,
// 			employees: 1000 + Math.random() * 200 + (year - 2020) * 150,
// 			employees2: 1000 + Math.random() * 200 + (year - 2020) * 150,
// 			employees3: 1000 + Math.random() * 200 + (year - 2020) * 150,
// 			marketShare: 15 + Math.random() * 5 + (year - 2020) * 1.2,
// 			profit: 20 + Math.random() * 15 + (year - 2020) * 5,
// 			assets: 500 + Math.random() * 100 + (year - 2020) * 80,
// 		});

// 		current.setMonth(current.getMonth() + 3);
// 	}

// 	return data;
// };

interface ChartProps {
	dataLegend: DataLegend[];
	data: Record<string, any>[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
	if (active && payload && payload.length) {
		return (
			<div className="rounded-lg border bg-background p-3 shadow-md">
				<p className="mb-2 font-medium">{label}</p>
				{payload.map((entry: any) => (
					<p className="flex items-center gap-2 text-sm" key={entry.name}>
						<span
							className="h-3 w-3 rounded-full"
							style={{ backgroundColor: entry.color }}
						/>
						<span className="text-muted-foreground">{entry.name}:</span>
						<span className="font-medium">{entry.value.toFixed(2)}</span>
					</p>
				))}
			</div>
		);
	}
	return null;
};

export function Chart({ dataLegend, data }: ChartProps) {
	// const data = useMemo(
	// 	() => generateData(dateRange.start, dateRange.end),
	// 	[dateRange],
	// );

	return (
		<div className="flex flex-col items-center justify-center pt-8">
			<ResponsiveContainer height={600} width="100%">
				<AreaChart
					accessibilityLayer
					data={data}
					margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
				>
					<CartesianGrid className="stroke-stone-300" strokeDasharray="3 3" />
					<XAxis
						className="text-xs"
						dataKey="date"
						tick={{ fill: "hsl(var(--muted-foreground))" }}
					/>
					<YAxis
						className="text-xs"
						// domain={[-5, "auto"]}
						padding={{ bottom: 60 }}
						tick={{ fill: "hsl(var(--muted-foreground))" }}
					/>
					{/*<Area
							dataKey="mobile"
							type="natural"
							fill="url(#fillMobile)"
							fillOpacity={0.4}
							stroke="var(--color-mobile)"
							stackId="a"
						/>*/}

					<Tooltip content={<CustomTooltip />} />
					<defs>
						{Array.from({ length: 10 }).map((_, index) => (
							<linearGradient
								id={`chart-grad-${index + 1}`}
								key={index}
								x1="0"
								x2="0"
								y1="0"
								y2="1"
							>
								<stop
									offset="5%"
									stopColor={`var(--chart-${index + 1})`}
									stopOpacity={0.6}
								/>
								<stop
									offset="65%"
									stopColor={`var(--chart-${index + 1})`}
									stopOpacity={0.01}
								/>
							</linearGradient>
						))}
					</defs>
					{/*<Legend wrapperStyle={{ paddingTop: "20px" }} />*/}
					{dataLegend.map((item, index) => (
						// <Area
						//   dataKey={item.label}
						//   dot={true}
						//   fill={`url(#chart-grad-${(index % 10) + 1})`}
						//   fillOpacity={0.4}
						//   key={item.id}
						//   stackId="a"
						//   stroke={`var(--chart-${(index % 10) + 1})`}
						//   type="natural"
						// />
						<Line
							key={item.id}
							type="monotone"
							dataKey={item.label}
							stroke={`var(--chart-${(index % 10) + 1})`}
							name={item.label}
							strokeWidth={2}
							dot={true}
							activeDot={{ r: 4 }}
						/>
					))}
				</AreaChart>
			</ResponsiveContainer>
		</div>
	);
}
