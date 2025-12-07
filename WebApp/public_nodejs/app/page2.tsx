"use client";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@bluevoid-test/ui/card";
import { Checkbox } from "@bluevoid-test/ui/checkbox";
import { Label } from "@bluevoid-test/ui/label";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { orpc } from "@/lib/orpc";
import { Chart } from "./_components/Chart";

export type DataLegend = {
	id: string;
	label: string;
	enabled: boolean;
};

const legendX: DataLegend[] = [
	// { id: "numer PKD", label: "numer PKD",  enabled: true },
	// { id: "nazwa PKD", label: "nazwa PKD",  enabled: true },
	// { id: "wskaźnik", label: "wskaźnik",  enabled: true },
	{ id: "2005", label: "2005", enabled: true },
	{ id: "2006", label: "2006", enabled: true },
	{ id: "2007", label: "2007", enabled: true },
	{ id: "2008", label: "2008", enabled: true },
	{ id: "2009", label: "2009", enabled: true },
	{ id: "2010", label: "2010", enabled: true },
	{ id: "2011", label: "2011", enabled: true },
	{ id: "2012", label: "2012", enabled: true },
	{ id: "2013", label: "2013", enabled: true },
	{ id: "2014", label: "2014", enabled: true },
	{ id: "2015", label: "2015", enabled: true },
	{ id: "2016", label: "2016", enabled: true },
	{ id: "2017", label: "2017", enabled: true },
	{ id: "2018", label: "2018", enabled: true },
	{ id: "2019", label: "2019", enabled: true },
	{ id: "2020", label: "2020", enabled: true },
	{ id: "2021", label: "2021", enabled: true },
	{ id: "2022", label: "2022", enabled: true },
	{ id: "2023", label: "2023", enabled: true },
	{ id: "2024", label: "2024", enabled: true },
];

const legendY: DataLegend[] = [
	{ id: "OG", label: "OG", enabled: true },
	// { id: "OG", label: "OG" ,enabled:true},
	{ id: "SEK_A", label: "SEK_A", enabled: true },
	// { id: "SEK_A", label: "SEK_A" ,enabled:true},
	{ id: "SEK_B", label: "SEK_B", enabled: true },
	// { id: "SEK_B", label: "SEK_B" ,enabled:true},
	{ id: "SEK_C", label: "SEK_C", enabled: true },
	// { id: "SEK_C", label: "SEK_C" ,enabled:true},
	{ id: "SEK_D", label: "SEK_D", enabled: true },
	// { id: "SEK_D", label: "SEK_D" ,enabled:true},
	{ id: "SEK_E", label: "SEK_E", enabled: true },
	// { id: "SEK_E", label: "SEK_E" ,enabled:true},
	{ id: "SEK_F", label: "SEK_F", enabled: true },
	// { id: "SEK_F", label: "SEK_F" ,enabled:true},
	{ id: "SEK_G", label: "SEK_G", enabled: true },
	// { id: "SEK_G", label: "SEK_G" ,enabled:true},
	{ id: "SEK_H", label: "SEK_H", enabled: true },
	// { id: "SEK_H", label: "SEK_H" ,enabled:true},
	{ id: "SEK_I", label: "SEK_I", enabled: true },
	// { id: "SEK_I", label: "SEK_I" ,enabled:true},
	{ id: "SEK_J", label: "SEK_J", enabled: true },
	// { id: "SEK_J", label: "SEK_J" ,enabled:true},
	{ id: "SEK_K", label: "SEK_K", enabled: true },
	// { id: "SEK_K", label: "SEK_K" ,enabled:true},
	{ id: "SEK_L", label: "SEK_L", enabled: true },
	// { id: "SEK_L", label: "SEK_L" ,enabled:true},
	{ id: "SEK_M", label: "SEK_M", enabled: true },
	// { id: "SEK_M", label: "SEK_M" ,enabled:true},
	{ id: "SEK_N", label: "SEK_N", enabled: true },
	// { id: "SEK_N", label: "SEK_N" ,enabled:true},
	{ id: "SEK_O", label: "SEK_O", enabled: true },
	// { id: "SEK_O", label: "SEK_O" ,enabled:true},
	{ id: "SEK_P", label: "SEK_P", enabled: true },
	// { id: "SEK_P", label: "SEK_P" ,enabled:true},
	{ id: "SEK_Q", label: "SEK_Q", enabled: true },
	// { id: "SEK_Q", label: "SEK_Q" ,enabled:true},
	{ id: "SEK_R", label: "SEK_R", enabled: true },
	// { id: "SEK_R", label: "SEK_R" ,enabled:true},
	{ id: "SEK_S", label: "SEK_S", enabled: true },
	// { id: "SEK_S", label: "SEK_S" ,enabled:true},
	{ id: "SEK_T", label: "SEK_T", enabled: true },
	// { id: "SEK_T", label: "SEK_T" ,enabled:true},
	{ id: "SEK_U", label: "SEK_U", enabled: true },
	// { id: "SEK_U", label: "SEK_U" ,enabled:true},
];

export function GrowthDashboard() {
	const [level, setLevel] = useState<"sektor" | "group">("group");

	const [legend, setLegend] = useState<DataLegend[]>(legendY);
	const { data: dataObj } = useQuery(orpc.testLoader.queryOptions());

	const enabledLabels = useMemo(
		() => legend.filter((f) => f.enabled),
		[legend],
	);
	let data: any = null;
	if (dataObj) data = dataObj.data;
	const newData = useMemo(
		() =>
			data?.filter(
				(val: any) => val.type !== "EN Liczba jednostek gospodarczych",
			),
		[data],
	);
	if (!dataObj) {
		return null;
	}

	const toggleItem = (id: string) => {
		setLegend((items) =>
			items.map((f) => (f.id === id ? { ...f, enabled: !f.enabled } : f)),
		);
	};

	return (
		<div className="flex min-h-[calc(100vh-3.5rem)] flex-col bg-background lg:flex-row">
			<div className="flex-1 p-4 lg:p-6">
        <Chart data={newData} dataLegend={enabledLabels} key={ `${}`} />
			</div>

			<div className="w-full border-border border-l bg-blue-100 lg:w-80 xl:w-96">
				<div className="space-y-6 p-4 lg:p-6">
					<Card>
						<CardHeader>
							<CardTitle className="text-base">Filtry PKD</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="space-y-4">
								{legend.map((item, index) => (
									<div className="flex items-center space-x-3" key={item.id}>
										<Checkbox
											checked={item.enabled}
											id={item.id}
											onCheckedChange={() => toggleItem(item.id)}
										/>
										<Label
											className="flex cursor-pointer items-center gap-2 font-normal"
											htmlFor={item.id}
										>
											<div
												className="h-3 w-3 rounded-full"
												style={{
													backgroundColor: `var(--chart-${(index % 10) + 1})`,
												}}
											/>
											{item.label}
										</Label>
									</div>
								))}
							</div>
						</CardContent>
					</Card>
				</div>
			</div>
		</div>
	);
}

export default GrowthDashboard;
