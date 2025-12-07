/** biome-ignore-all lint/correctness/useUniqueElementIds: This is test */

import { Checkbox } from "@bluevoid/ui/checkbox";
import { Label } from "@bluevoid/ui/label";

function TestCheckbox() {
	return (
		<>
			<div className="flex items-center gap-2">
				<Checkbox id="checkbox_unchecked" />
				<Label htmlFor="checkbox_unchecked">Unchecked</Label>
			</div>
			<div className="flex items-center gap-2">
				<Checkbox id="checkbox_checked" checked />
				<Label htmlFor="checkbox_checked">Checked</Label>
			</div>
		</>
	);
}

export default TestCheckbox;
