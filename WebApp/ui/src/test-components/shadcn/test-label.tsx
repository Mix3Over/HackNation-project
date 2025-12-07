import { Checkbox } from "@bluevoid/ui/checkbox";
import { Input } from "@bluevoid/ui/input";
import { Label } from "@bluevoid/ui/label";
import { useId } from "react";

function TestLabel() {
	const uuid = useId();
	return (
		<>
			<div className="grid w-full max-w-sm items-center gap-1.5">
				<Label htmlFor="picture">Picture</Label>
				<Input id={`${uuid}_picture`} type="file" />
			</div>
			<div className="flex items-center gap-2">
				<Checkbox id={`${uuid}_checkbox_unchecked`} />
				<Label htmlFor="checkbox_unchecked">Unchecked</Label>
			</div>
		</>
	);
}

export default TestLabel;
