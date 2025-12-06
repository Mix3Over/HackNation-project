import { Button } from "@bluevoid/ui/button";
import { Input } from "@bluevoid/ui/input";
import { Label } from "@bluevoid/ui/label";

function TestInput() {
	return (
		<>
			<div>
				<Input type="email" placeholder="Email" />
			</div>
			<div className="grid w-full max-w-sm items-center gap-1.5">
				<Label htmlFor="picture">Picture</Label>
				<Input id="picture" type="file" />
			</div>
			<div>
				<Input disabled type="email" placeholder="Email" />
			</div>

			<div className="flex w-full max-w-sm items-center space-x-2">
				<Input type="email" placeholder="Email" />
				<Button type="submit">Subscribe</Button>
			</div>
		</>
	);
}

export default TestInput;
