/** biome-ignore-all lint/correctness/useUniqueElementIds: This is test */

import { Label } from "@bluevoid/ui/label";
import { Textarea } from "@bluevoid/ui/textarea";

function TestTextarea() {
	return (
		<>
			<Textarea placeholder="Type your message here." />
			<Textarea placeholder="Type your message here." disabled />
			<div className="grid w-full gap-1.5">
				<Label htmlFor="message-2">Your Message</Label>
				<Textarea placeholder="Type your message here." id="message-2" />
				<p className="text-muted-foreground text-sm">
					Your message will be copied to the support team.
				</p>
			</div>
		</>
	);
}

export default TestTextarea;
